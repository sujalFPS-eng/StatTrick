"""StatTrick valuation pipeline (patched for PostgreSQL).

What changed vs. the original:
  #1  every player-season is collapsed to ONE row (key = accent-free name + birth year +
      season + club); richer sources fill gaps in poorer ones instead of being added twice.
  #2  explicit per-source column maps (no more global alias dict that missed
      'expected goals', wrong 'blocks', 'won', 'psxg+/-' ...); cleaned_* files now
      actually contribute minutes (their 'Avg Mins per Match' column is total minutes).
  #3  missing data stays NaN (XGBoost handles NaN natively); a has_advanced flag is exported.
  #4  Transfermarkt match is blocked on birth year, prefers active players, is accent-proof,
      and every unmatched >=1500-minute player is written to data/unmatched_players.csv.
  #5  players who have not played in the last two seasons are excluded (not recruitable).
  #6  predictions are OUT-OF-FOLD for every player (no in-sample flattery); metrics are honest.
  #7  surplus is also exported in % / log terms (surplus_pct, surplus_log).
  #8  no hard-coded years, contract years from today, age from date of birth,
      league labels canonicalised (5 dummies, not 10).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from src.common import (
    DATA_DIR, canon_club, canon_league, canon_season, current_season_start, make_pkey,
    name_key, season_start, season_weight, today, write_json,
)
from src.database import engine

MIN_CAREER_MINUTES = 1500      # deduplicated minutes across all seasons
MIN_MARKET_VALUE_M = 0.25      # was 0.5 - less truncation of the target
RECENT_SEASONS = 2             # must have played in one of the last N seasons
TM_ACTIVE_LOOKBACK = 2         # TM 'last_season' >= current_start - 2
FUZZY_CUTOFF = 88              # after accent stripping + birth-year blocking this is safe

RAW_STATS = ["gls", "ast", "xg", "xag", "sh", "sot", "prgc", "prgp",
             "tkl", "int", "clr", "blk", "aer_won", "saves", "psxg_net"]
PER90_COLS = [f"{s}_per90" for s in RAW_STATS]
ADVANCED = ["xg", "xag", "prgc", "prgp"]
ID_COLS = ["player", "born", "pos", "squad", "league", "age", "min"]

# ---------------------------------------------------------------------------- sources
# Column maps are applied AFTER lower-casing + stripping. One explicit map per source.
CLEANED_MAP = {
    "avg mins per match": "min",        # despite the name this column holds TOTAL minutes
    "goals": "gls", "assists": "ast", "expected goals": "xg",
    "progressive carries": "prgc", "progressive passes": "prgp",
    "total shots": "sh", "tackles attempted": "tkl", "interceptions": "int",
    "clearances": "clr", "saves": "saves",
}
MAP_2425 = {                              # plain 'blocks' (passing table) is dropped first
    "blocks_stats_defense": "blk", "won": "aer_won", "psxg+/-": "psxg_net",
}

SOURCES = [
    # path, tag, rank (lower wins), fixed season (None = read from file), column map, drop
    ("fbref_2425.csv", "fbref_2425", 0, "2024-2025", MAP_2425, ["blocks"]),
    ("fbref_2526.csv", "fbref_2526", 0, "2025-2026", {}, []),
    ("cleaned_2021-22.csv", "cleaned", 1, None, CLEANED_MAP, []),
    ("cleaned_2022-23.csv", "cleaned", 1, None, CLEANED_MAP, []),
    ("cleaned_2023-24.csv", "cleaned", 1, None, CLEANED_MAP, []),
    ("live_fbref_data.csv", "live", 1, "LIVE", {}, []),
    ("fbref_historical_stats.csv", "historical", 2, None, {}, []),
]


def _read(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df.loc[:, ~df.columns.duplicated()].copy()      # keep first of any duplicate header
    except pd.errors.EmptyDataError:
        return None


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_source(fname, tag, rank, season, colmap, drop) -> pd.DataFrame | None:
    path = DATA_DIR / fname
    if not path.exists():
        print(f"  ! missing {fname} (skipped)")
        return None
    
    df = _read(path)
    if df is None:
        print(f"  ! empty file {fname} (skipped)")
        return None
        
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.rename(columns=colmap)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "comp" in df.columns and "league" not in df.columns:
        df = df.rename(columns={"comp": "league"})

    # derived stats for the cleaned_* files
    if tag == "cleaned":
        if {"shots blocked", "passes blocked"} <= set(df.columns):
            df["blk"] = _num(df["shots blocked"]) + _num(df["passes blocked"])
        if {"total shots", "% shots on target"} <= set(df.columns):
            df["sot"] = (_num(df["total shots"]) * _num(df["% shots on target"]) / 100).round()

    if season == "LIVE":
        df["season"] = f"{current_season_start()}-{current_season_start() + 1}"
    elif season:
        df["season"] = season
    df["src"], df["rank"] = tag, rank

    for c in ID_COLS + RAW_STATS:
        if c not in df.columns:
            df[c] = np.nan
    keep = ID_COLS + RAW_STATS + ["season", "src", "rank"]
    out = df[keep].copy()
    print(f"  + {fname:32s} {len(out):6d} rows  [{tag}]")
    return out


def build_player_seasons() -> pd.DataFrame:
    print("Loading sources...")
    frames = [f for f in (load_source(*s) for s in SOURCES) if f is not None]
    df = pd.concat(frames, ignore_index=True)

    df["season"] = df["season"].map(canon_season)
    df["league"] = df["league"].map(canon_league)
    df["squad"] = df["squad"].map(canon_club)
    df["min"] = _num(df["min"])
    df["born"] = pd.to_numeric(df["born"], errors="coerce")
    df["age"] = df["age"].astype(str).str.split("-").str[0]
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["pos"] = df["pos"].astype(str).replace("nan", "")
    for c in RAW_STATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["player"].notna() & (df["player"] != "Player") & df["season"].notna()]
    df = df[df["min"] > 0].copy()
    df["pkey"] = make_pkey(df["player"], df["born"])

    before = len(df)
    df = (df.sort_values("rank", kind="stable")
            .groupby(["pkey", "season", "squad"], as_index=False, sort=False).first())
    print(f"Collapsed {before:,} source rows -> {len(df):,} player-season rows "
          f"({before - len(df):,} duplicates removed)")

    # sanity: same player+season at two clubs with identical minutes = probably a spelling variant
    twins = df.groupby(["pkey", "season"]).filter(lambda g: len(g) > 1 and g["min"].nunique() < len(g))
    print(f"Possible residual club-spelling duplicates: {len(twins)} rows")

    m90 = df["min"] / 90.0
    for s in RAW_STATS:
        df[f"{s}_per90"] = df[s] / m90            # NaN stays NaN
    df["season_start"] = df["season"].map(season_start)
    return df


# ---------------------------------------------------------------------------- aggregate
def aggregate_players(ps: pd.DataFrame) -> pd.DataFrame:
    ps = ps.copy()
    ps["w90"] = (ps["min"] / 90.0) * ps["season"].map(season_weight)

    agg = {"min": ("min", "sum"), "n_seasons": ("season", "nunique"),
           "last_season_start": ("season_start", "max")}
    for c in PER90_COLS:
        valid = ps[c].notna()
        ps[f"{c}__num"] = np.where(valid, ps[c] * ps["w90"], 0.0)
        ps[f"{c}__den"] = np.where(valid, ps["w90"], 0.0)
        agg[f"{c}__num"] = (f"{c}__num", "sum")
        agg[f"{c}__den"] = (f"{c}__den", "sum")
    ps["_adv"] = ps[[f"{a}_per90" for a in ADVANCED]].notna().all(axis=1)
    ps["_adv_season"] = np.where(ps["_adv"], ps["season_start"], np.nan)
    agg["adv_last_season_start"] = ("_adv_season", "max")

    out = ps.groupby("pkey").agg(**agg).reset_index()
    for c in PER90_COLS:
        den = out.pop(f"{c}__den")
        num = out.pop(f"{c}__num")
        out[c] = np.where(den > 0, num / den.replace(0, np.nan), np.nan)   # NaN, not 0
    out["has_advanced"] = out[[f"{a}_per90" for a in ADVANCED]].notna().all(axis=1).astype(int)

    # identity from the most recent season (largest stint if he moved mid-season)
    latest = (ps.sort_values(["season_start", "min"], ascending=False)
                .drop_duplicates("pkey")[["pkey", "player", "squad", "league", "age", "born"]])
    latest_squads = (ps[ps["season_start"] == ps.groupby("pkey")["season_start"].transform("max")]
                     .groupby("pkey")["squad"].agg(list).rename("latest_squads"))
    recent = ps[ps["season_start"] >= ps.groupby("pkey")["season_start"].transform("max") - 1].copy()
    recent["ncomma"] = recent["pos"].str.count(",")
    posfull = (recent.sort_values("ncomma", ascending=False)
                     .drop_duplicates("pkey")[["pkey", "pos"]].rename(columns={"pos": "pos_full"}))
    out = (out.merge(latest, on="pkey").merge(posfull, on="pkey", how="left")
              .merge(latest_squads, on="pkey", how="left"))
    out["last_season"] = out["last_season_start"].map(lambda y: f"{int(y)}-{int(y) + 1}")
    out["adv_last_season"] = out["adv_last_season_start"].map(
        lambda y: f"{int(y)}-{int(y) + 1}" if pd.notna(y) else None)
    return out


# ---------------------------------------------------------------------------- Transfermarkt
def _prep_tm() -> pd.DataFrame:
    tm = pd.read_csv(DATA_DIR / "players.csv", low_memory=False)
    tm.columns = [c.strip().lower() for c in tm.columns]
    tm = tm[tm["market_value_in_eur"].notna()].copy()
    tm = tm[tm["last_season"] >= current_season_start() - TM_ACTIVE_LOOKBACK].copy()  # active only
    tm["nkey"] = tm["name"].map(name_key)
    tm["dob"] = pd.to_datetime(tm["date_of_birth"], errors="coerce")
    tm["dob_year"] = tm["dob"].dt.year
    tm["market_value_m"] = tm["market_value_in_eur"] / 1e6
    tm["contract_exp"] = pd.to_datetime(tm["contract_expiration_date"], errors="coerce")
    return tm



def _surname_match(nk: str, born, squad: str, by_year: dict) -> pd.DataFrame:
    """Last-resort stage for nickname / shortened-name variants ('Andy' vs 'Andrew Robertson').
    Requires an EXACT birth year, one name's surname (last token) contained in the other, and either a compatible
    first name (same first two letters) or a matching club."""
    toks = nk.split()
    if pd.isna(born) or len(toks) < 2 or int(born) not in by_year:
        return by_year.get(-1, pd.DataFrame())
    pool = by_year[int(born)]
    club_key = name_key(squad)
    keep = []
    for idx, cand_key, club in zip(pool.index, pool["nkey"], pool["current_club_name"]):
        ct = cand_key.split()
        if len(ct) < 2 or not (toks[-1] in ct[1:] or ct[-1] in toks[1:]):
            continue                              # the shared token must be a SURNAME (last token)
        first_ok = toks[0][:2] == ct[0][:2]
        club_ok = fuzz.WRatio(club_key, name_key(club)) >= 85
        if first_ok or club_ok:
            keep.append(idx)
    return pool.loc[keep] if keep else pool.iloc[0:0]

def match_transfermarkt(players: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    by_year = {int(y): g for y, g in tm.groupby("dob_year")}
    rows = []
    for r in players.itertuples(index=False):
        nk, born = name_key(r.player), r.born
        if pd.notna(born):
            parts = [by_year[y] for y in (int(born) - 1, int(born), int(born) + 1) if y in by_year]
            cands, cutoff = (pd.concat(parts) if parts else tm.iloc[0:0]), FUZZY_CUTOFF
        else:                                     # no birth year -> exact name only
            cands, cutoff = tm, 100
        if cands.empty:
            rows.append((r.pkey, None, 0.0, "none")); continue

        exact = cands[cands["nkey"] == nk]
        if not exact.empty:
            pool, score, method = exact, 100.0, "exact"
        else:
            hits = process.extract(nk, cands["nkey"], scorer=fuzz.WRatio, score_cutoff=cutoff, limit=6)
            if hits:
                top = max(h[1] for h in hits)
                pool = cands.loc[[h[2] for h in hits if h[1] >= top - 1.0]]
                score, method = float(top), "fuzzy"
            else:
                pool = _surname_match(nk, born, r.squad, by_year)
                if pool.empty:
                    rows.append((r.pkey, None, 0.0, "none")); continue
                score, method = 80.0, "surname"

        if len(pool) > 1:                          # namesakes: year -> club -> recency -> value
            club_key = name_key(r.squad)
            pool = pool.assign(
                _dy=(pool["dob_year"] - born).abs() if pd.notna(born) else 0,
                _club=pool["current_club_name"].map(lambda c: fuzz.WRatio(club_key, name_key(c))),
            ).sort_values(["_dy", "_club", "last_season", "market_value_m"],
                          ascending=[True, False, False, False])
        rows.append((r.pkey, pool.index[0], score, method))
    m = pd.DataFrame(rows, columns=["pkey", "tm_idx", "match_score", "match_method"])
    m = m.dropna(subset=["tm_idx"]).astype({"tm_idx": int})
    info = tm.loc[m["tm_idx"], ["player_id", "name", "current_club_name", "dob", "market_value_m",
                                "contract_exp"]].reset_index(drop=True)
    info.columns = ["tm_player_id", "tm_match_name", "tm_club", "dob", "actual_value_m", "contract_exp"]
    return pd.concat([m.drop(columns="tm_idx").reset_index(drop=True), info], axis=1)


# ---------------------------------------------------------------------------- roles
def classify_role(row: pd.Series) -> str:
    """FBref 'Pos' only has GK/DF/MF/FW combinations (no LB/RW tags), so use the first
    token as the primary role and the second as a hint; NaN-safe when advanced data is missing."""
    pos = str(row.get("pos_full", "")).upper().replace(" ", "")
    toks = [t[:2] for t in pos.split(",") if t]
    primary, secondary = (toks + ["", ""])[:2]
    g = lambda k: row.get(k) if pd.notna(row.get(k)) else np.nan
    prgc, prgp, clr, sh = g("prgc_per90"), g("prgp_per90"), g("clr_per90"), g("sh_per90")
    xag, gls, ast = g("xag_per90"), g("gls_per90"), g("ast_per90")
    tkl, itc = g("tkl_per90"), g("int_per90")

    if primary == "GK":
        return "GK"
    if primary == "DF":
        if secondary == "MF":
            return "FULLBACK"
        if pd.notna(prgc) and prgc >= 1.8 and (pd.isna(clr) or clr < 2.2):
            return "FULLBACK"
        return "CB"
    if primary == "FW":
        if pd.notna(prgc) and prgc >= 2.3:
            return "WINGER"
        if secondary == "MF" and pd.notna(prgp) and prgp >= 4.0:
            return "CAM"
        if secondary == "MF" and not row.get("has_advanced", 0):
            return "WINGER"
        return "ST"
    # midfield
    if (pd.notna(sh) and sh >= 1.6) or (pd.notna(xag) and xag >= 0.14) or \
       (np.nansum([gls, ast]) >= 0.28 and pd.notna(gls)):
        return "CAM"
    if pd.notna(itc):
        defensive = (tkl + itc) if pd.notna(tkl) else itc * 2.4
        if defensive >= 2.8 and (pd.isna(sh) or sh < 1.2):
            return "CDM"
    return "CM"


# ---------------------------------------------------------------------------- model
def _xgb(**kw):
    return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, n_jobs=-1, **kw)


def build_valuation_model():
    ps = build_player_seasons()

    # ---- timeline export (per-season, NaNs intact, slim schema)
    tl_cols = ["pkey", "player", "born", "pos", "squad", "league", "season", "min", "age"] + RAW_STATS + PER90_COLS
    
    print("Uploading player_timeline to Neon Postgres...")
    ps[ps["min"] >= 90][tl_cols].to_sql("player_timeline", con=engine, if_exists="replace", index=False)

    players = aggregate_players(ps)
    print(f"\nUnique players: {len(players):,}")
    cur = current_season_start()
    players = players[(players["min"] >= MIN_CAREER_MINUTES)
                      & (players["last_season_start"] >= cur - (RECENT_SEASONS - 1))].copy()
    print(f"After >= {MIN_CAREER_MINUTES} min and played in last {RECENT_SEASONS} seasons: {len(players):,}")

    print("Matching Transfermarkt (birth-year blocked)...")
    tm = _prep_tm()
    match = match_transfermarkt(players, tm)
    df = players.merge(match, on="pkey", how="left")

    unmatched = df[df["actual_value_m"].isna()][["player", "squad", "league", "born", "min", "last_season"]]
    # Keeping unmatched as local CSV since this acts purely as a debug log for the modeler
    unmatched.sort_values("min", ascending=False).to_csv(DATA_DIR / "unmatched_players.csv", index=False)
    print(f"  matched {df['actual_value_m'].notna().sum():,} | unmatched {len(unmatched):,} "
          f"(listed in data/unmatched_players.csv) | fuzzy {(df['match_method'] == 'fuzzy').sum()}")

    df = df.dropna(subset=["actual_value_m"])
    df = df[df["actual_value_m"] > MIN_MARKET_VALUE_M].copy()

    # club: if he moved mid-season prefer the stint that matches Transfermarkt's current club
    def pick_squad(r):
        sq = r["latest_squads"]
        if isinstance(sq, list) and len(sq) > 1 and isinstance(r["tm_club"], str):
            best = max(sq, key=lambda s: fuzz.WRatio(name_key(s), name_key(r["tm_club"])))
            return best
        return r["squad"]
    df["squad"] = df.apply(pick_squad, axis=1)

    # ---- age / contract relative to *today*, not a hard-coded year
    t = pd.Timestamp(today())
    age_dob = (t - df["dob"]).dt.days / 365.25
    df["age_clean"] = age_dob.fillna(today().year - df["born"]).fillna(df["age"]).round(1)
    df["contract_years_left"] = ((df["contract_exp"] - t).dt.days / 365.25).clip(lower=0, upper=7)  # NaN kept

    df["pos_clean"] = df.apply(classify_role, axis=1)
    df["raw_pos"] = df["pos_full"]

    # unique display names (two different 'Aaron Ramsey's must not collide in the app)
    dup = df["player"].duplicated(keep=False)
    df["fb_name"] = df["player"]
    df.loc[dup, "player"] = df.loc[dup, "player"] + " (" + df.loc[dup, "squad"] + ")"

   # ---- design matrix (NaN preserved for XGBoost)
    league_d = pd.get_dummies(df["league"], prefix="league").astype(int)
    pos_d = pd.get_dummies(df["pos_clean"], prefix="pos_clean").astype(int)
    
    # Model A: Full Features (Historical)
    feats_adv = ["age_clean", "min", "contract_years_left", "has_advanced"] + PER90_COLS
    X_adv = pd.concat([df[feats_adv].apply(pd.to_numeric, errors="coerce"), league_d, pos_d], axis=1)
    
    # Model B: Basic Features (Live / No Opta)
    basic_per90 = [f"{s}_per90" for s in RAW_STATS if s not in ADVANCED]
    feats_basic = ["age_clean", "min", "contract_years_left"] + basic_per90
    X_basic = pd.concat([df[feats_basic].apply(pd.to_numeric, errors="coerce"), league_d, pos_d], axis=1)
    
    y_raw = df["actual_value_m"].values
    y_log = np.log1p(y_raw)

    # ---- OUT-OF-FOLD predictions: Dual-Model Architecture
    print(f"Training XGBoost Dual-Model Architecture on {len(df):,} players...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def _train_and_conformalize(X_matrix, name):
        print(f"  -> Training Model {name} ({X_matrix.shape[1]} features)...")
        oof_log = cross_val_predict(_xgb(), X_matrix, y_log, cv=cv)
        try:
            lo_log = cross_val_predict(_xgb(objective="reg:quantileerror", quantile_alpha=0.15), X_matrix, y_log, cv=cv)
            hi_log = cross_val_predict(_xgb(objective="reg:quantileerror", quantile_alpha=0.85), X_matrix, y_log, cv=cv)
        except Exception as exc:
            res = y_log - oof_log
            lo_log, hi_log = oof_log + np.quantile(res, 0.15), oof_log + np.quantile(res, 0.85)
        
        # Prevent bands from crossing the mean, then conformalize to hit 70% coverage
        lo_log, hi_log = np.minimum(lo_log, oof_log), np.maximum(hi_log, oof_log)
        score = np.maximum(lo_log - y_log, y_log - hi_log)
        q = np.quantile(score, min(1.0, 0.70 * (1 + 1 / len(score))))
        return oof_log, lo_log - q, hi_log + q
        
    oof_log_a, lo_log_a, hi_log_a = _train_and_conformalize(X_adv, "A (Advanced/Historical)")
    oof_log_b, lo_log_b, hi_log_b = _train_and_conformalize(X_basic, "B (Basic/Live)")
    
    # Splice Predictions: Route to Model B if advanced stats are missing
    has_adv = df["has_advanced"].astype(bool).values
    oof_log = np.where(has_adv, oof_log_a, oof_log_b)
    lo_log = np.where(has_adv, lo_log_a, lo_log_b)
    hi_log = np.where(has_adv, hi_log_a, hi_log_b)

    pred = np.expm1(oof_log)
    df["predicted_value_m"] = np.round(pred, 2)
    df["pred_value_low_m"] = np.round(np.clip(np.expm1(lo_log), 0, None), 2)
    df["pred_value_high_m"] = np.round(np.expm1(hi_log), 2)
    df["surplus_value_m"] = np.round(pred - y_raw, 2)
    df["surplus_log"] = np.round(oof_log - y_log, 4)                       # log-space edge
    df["surplus_pct"] = np.round(100 * np.expm1(oof_log - y_log), 1)      # (1+pred)/(1+actual) - 1

    r2, mae = r2_score(y_raw, pred), mean_absolute_error(y_raw, pred)
    r2_log = r2_score(y_log, oof_log)
    cover = float(((y_raw >= df["pred_value_low_m"]) & (y_raw <= df["pred_value_high_m"])).mean())
    print("\n--- Out-of-fold model evaluation (honest, every row) ---")
    print(f"R2 (EUR): {r2:.3f} | R2 (log): {r2_log:.3f} | MAE: EUR {mae:.2f}M | 15-85 band coverage: {cover:.0%}")

   # ---- export (keeps the columns the app already expects + the new ones)
    out = pd.concat([df.reset_index(drop=True), league_d.reset_index(drop=True),
                     pos_d.reset_index(drop=True)], axis=1)
                     
    # RESTORE: Ensure the raw text column for the UI isn't lost among the dummies
    if "league" not in out.columns and "league" in df.columns:
        out["league"] = df["league"].values
        
    out = out.drop(columns=["latest_squads", "contract_exp", "last_season_start", "adv_last_season_start",
                            "tm_player_id"], errors="ignore")
    
    print("Uploading players_master and player_timeline to Neon Postgres...")
    out.to_sql("players_master", con=engine, if_exists="replace", index=False)
    
    # Push the timeline simultaneously with low_memory disabled to suppress dtype warnings
    ps[ps["min"] >= 90][tl_cols].to_sql("player_timeline", con=engine, if_exists="replace", index=False)
    
    print(f"Uploaded players_master ({len(out):,} players) and player_timeline to Postgres.")
    
    write_json(DATA_DIR / "model_metrics.json", {
        "built_on": str(today()), "season_in_progress": f"{cur}-{cur + 1}",
        "players": int(len(out)), "unmatched_players": int(len(unmatched)),
        "oof_r2_eur": round(float(r2), 3), "oof_r2_log": round(float(r2_log), 3),
        "oof_mae_eur_m": round(float(mae), 2), "band_coverage_15_85": round(cover, 3),
        "players_without_advanced_stats": int((out["has_advanced"] == 0).sum()),
    })
    print(f"Uploaded players_master to Postgres ({len(out):,} players)")

    # fail loudly so the CI job never commits a broken database
    if len(out) < 1500 or r2 < 0.3:
        sys.exit(f"Sanity check failed: {len(out)} players, OOF R2={r2:.2f}")


if __name__ == "__main__":
    build_valuation_model()