"""Market forecast: where will Transfermarkt put this player's value in 12 months?

This is deliberately a DIFFERENT model from the performance value in valuation_model.py.

  performance value  - "what is this output worth?"  Never sees a Transfermarkt price.
  market forecast    - "what will the market say next year?"  Uses everything the market itself
                       reacts to, openly including today's Transfermarkt value, its recent
                       momentum, the player's peak, his age and his club's level, plus output.

Training data: one row per player per past June ("anchor"). Features are what was known on that
date; the target is the log change in his Transfermarkt value over the following 12 months.
Players who later left the top-5 leagues stay in the training set (Transfermarkt keeps valuing
them), so the model is not trained only on survivors.

Honesty check: the model is back-tested on the most recent completed year it never saw
(trained on earlier anchors only) and the result is exported with the forecast.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.common import DATA_DIR

HISTORY_FILE = DATA_DIR / "player_valuations.csv.gz"
MAX_VALUE_AGE_DAYS = 400       # a "current" value must be at most this old on the anchor date
FRESH_TARGET_DAYS = 150        # the target must be a valuation made well after the anchor
VALUE_FEATURES = ["lv0", "mom6", "mom12", "peak_ratio", "age", "club_lv", "posg"]
PERF_FEATURES = ["min", "ga90", "xg90", "sh90", "prev_min", "prev_ga90", "min_change"]
LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]


def _model() -> XGBRegressor:
    return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=5, random_state=42, n_jobs=-1)


def _asof(hist: pd.DataFrame, when: pd.Timestamp, max_age_days: int, after: pd.Timestamp | None = None):
    """Latest valuation per player on or before `when` (optionally strictly after `after`)."""
    x = hist[hist["date"] <= when]
    if after is not None:
        x = x[x["date"] > after]
    x = x.groupby("player_id").tail(1)
    x = x[x["date"] >= when - pd.Timedelta(days=max_age_days)]
    return x.set_index("player_id")


def _season_table(ps: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season with stints combined."""
    some = lambda s: s.sum(min_count=1)                     # noqa: E731 - NaN stays NaN
    return (ps.groupby(["pkey", "season_start"])
              .agg(min=("min", "sum"), gls=("gls", some), ast=("ast", some), xg=("xg", some),
                   sh=("sh", some), league=("league", "first"), pos=("pos", "first"))
              .reset_index())


def _anchor_rows(year: int, seasons: pd.DataFrame, ids: pd.DataFrame, hist: pd.DataFrame,
                 last_date: pd.Timestamp, with_target: bool) -> pd.DataFrame:
    when = min(pd.Timestamp(f"{year}-06-30"), last_date)
    v0 = _asof(hist, when, MAX_VALUE_AGE_DAYS)
    v6 = _asof(hist, when - pd.Timedelta(days=200), 600)
    v12 = _asof(hist, when - pd.Timedelta(days=365), 600)
    peak = hist[hist["date"] <= when].groupby("player_id")["market_value_in_eur"].max()

    s = seasons[seasons["season_start"] == year - 1].merge(ids, on="pkey")
    prev = (seasons[seasons["season_start"] == year - 2][["pkey", "min", "gls", "ast"]]
            .rename(columns={"min": "prev_min", "gls": "prev_gls", "ast": "prev_ast"}))
    s = s.merge(prev, on="pkey", how="left")
    pid = s["tm_player_id"]
    s["v0"], s["v0_date"] = pid.map(v0["market_value_in_eur"]), pid.map(v0["date"])
    s["club_id"] = pid.map(v0["current_club_id"])
    s["v6"], s["v12"], s["peak"] = pid.map(v6["market_value_in_eur"]), pid.map(v12["market_value_in_eur"]), pid.map(peak)
    if with_target:
        v1 = _asof(hist, pd.Timestamp(f"{year + 1}-06-30"), MAX_VALUE_AGE_DAYS,
                   after=when + pd.Timedelta(days=FRESH_TARGET_DAYS))
        s["v1"] = pid.map(v1["market_value_in_eur"])
    else:
        s["v1"] = np.nan
    s["age"] = (when - pd.to_datetime(s["dob"])).dt.days / 365.25
    s["anchor_year"] = year
    return s


def _features(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["v0"].notna() & (d["v0"] > 0)].copy()
    d["lv0"] = np.log(d["v0"])
    d["mom6"], d["mom12"] = np.log(d["v0"] / d["v6"]), np.log(d["v0"] / d["v12"])
    d["peak_ratio"] = d["v0"] / d["peak"]
    m90 = d["min"] / 90.0
    d["ga90"] = (d["gls"] + d["ast"]) / m90
    d["xg90"], d["sh90"] = d["xg"] / m90, d["sh"] / m90
    d["prev_ga90"] = (d["prev_gls"] + d["prev_ast"]) / (d["prev_min"] / 90.0)
    d["min_change"] = d["min"] - d["prev_min"]
    # club level on the anchor date = team-mates' mean log value (leave-one-out)
    grp = d.groupby(["anchor_year", "club_id"])["lv0"]
    n = grp.transform("size")
    d["club_lv"] = np.where(n > 1, (grp.transform("sum") - d["lv0"]) / (n - 1).clip(lower=1), np.nan)
    d["posg"] = d["pos"].astype(str).str[:2].map({"GK": 0, "DF": 1, "MF": 2, "FW": 3})
    for lg in LEAGUES:
        d[f"lg_{lg}"] = (d["league"] == lg).astype(int)
    d["y"] = np.log(d["v1"] / d["v0"])
    return d.replace([np.inf, -np.inf], np.nan)


def build_market_forecast(ps: pd.DataFrame, ids: pd.DataFrame):
    """ids: DataFrame[pkey, tm_player_id, dob]. Returns (per-pkey forecast frame, metrics dict,
    dated value history), or (None, {...reason}, None) when the history is unavailable or too thin."""
    if not HISTORY_FILE.exists():
        return None, {"available": False, "reason": "data/player_valuations.csv.gz not found "
                                                    "(run src/ingest_valuations.py)"}, None
    hist = pd.read_csv(HISTORY_FILE, usecols=["player_id", "date", "market_value_in_eur", "current_club_id"])
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date", "market_value_in_eur"]).sort_values(["player_id", "date"])
    ids = ids.dropna(subset=["tm_player_id"]).copy()
    ids["tm_player_id"] = ids["tm_player_id"].astype(int)
    hist = hist[hist["player_id"].isin(set(ids["tm_player_id"]))]

    last_date = hist["date"].max()
    predict_year = last_date.year if last_date.month >= 5 else last_date.year - 1
    seasons = _season_table(ps)
    have = set(seasons["season_start"])
    train_years = [y for y in range(predict_year - 6, predict_year) if (y - 1) in have]
    if len(train_years) < 3 or (predict_year - 1) not in have:
        return None, {"available": False, "reason": f"not enough history (anchors {train_years})"}, None

    data = _features(pd.concat(
        [_anchor_rows(y, seasons, ids, hist, last_date, with_target=True) for y in train_years]
        + [_anchor_rows(predict_year, seasons, ids, hist, last_date, with_target=False)], ignore_index=True))
    feats = VALUE_FEATURES + [f"lg_{lg}" for lg in LEAGUES] + PERF_FEATURES
    labelled = data[data["y"].notna() & (data["anchor_year"] < predict_year)]

    # ---- back-test: train on everything before the last completed year, test on that year
    test_year = max(train_years)
    tr, te = labelled[labelled["anchor_year"] < test_year], labelled[labelled["anchor_year"] == test_year]
    pred = _model().fit(tr[feats], tr["y"]).predict(te[feats])
    y = te["y"].to_numpy()
    resid = y - pred
    ss_tot = ((y - y.mean()) ** 2).sum()
    v0, v1 = te["v0"].to_numpy(), te["v1"].to_numpy()
    order = pd.qcut(pd.Series(pred).rank(method="first"), 5, labels=False).to_numpy()
    miss = np.abs(np.exp(pred - y) - 1)                    # forecast / actual - 1
    moved = y != 0
    metrics = {
        "available": True,
        "within_10pct": round(float((miss <= 0.10).mean()), 3),
        "within_25pct": round(float((miss <= 0.25).mean()), 3),
        "within_50pct": round(float((miss <= 0.50).mean()), 3),
        "median_miss_pct": round(float(np.median(miss) * 100), 1),
        "direction_right": round(float((np.sign(pred[moved]) == np.sign(y[moved])).mean()), 3),
        "no_change_within_25pct": round(float((np.abs(np.exp(-y) - 1) <= 0.25).mean()), 3),
        "trained_on_anchors": [int(a) for a in train_years],
        "backtest": f"{test_year} -> {test_year + 1}",
        "backtest_players": int(len(te)),
        "r2_of_change": round(float(1 - (resid ** 2).sum() / ss_tot), 3),
        "mae_eur_m": round(float(np.abs(v1 - v0 * np.exp(pred)).mean() / 1e6), 2),
        "mae_eur_m_if_no_change": round(float(np.abs(v1 - v0).mean() / 1e6), 2),
        "median_actual_change_pct": round(float((np.exp(np.median(y)) - 1) * 100), 1),
        "top_fifth_predicted_pct": round(float((np.exp(np.median(pred[order == 4])) - 1) * 100), 1),
        "top_fifth_actual_pct": round(float((np.exp(np.median(y[order == 4])) - 1) * 100), 1),
        "top_fifth_share_rose": round(float((y[order == 4] > 0).mean()), 2),
        "bottom_fifth_predicted_pct": round(float((np.exp(np.median(pred[order == 0])) - 1) * 100), 1),
        "bottom_fifth_actual_pct": round(float((np.exp(np.median(y[order == 0])) - 1) * 100), 1),
        "value_as_of": str(last_date.date()),
        "forecast_for": f"{predict_year + 1}-06",
    }
    lo_q, hi_q = np.quantile(resid, 0.15), np.quantile(resid, 0.85)      # 70% range of back-test misses

    # ---- final model on every labelled anchor, applied to the latest one
    final = _model().fit(labelled[feats], labelled["y"])
    now = data[data["anchor_year"] == predict_year].copy()
    p = final.predict(now[feats])
    out = pd.DataFrame({
        "pkey": now["pkey"].to_numpy(),
        "forecast_base_value_m": np.round(now["v0"].to_numpy() / 1e6, 2),
        "forecast_base_date": now["v0_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "forecast_value_m": np.round(now["v0"].to_numpy() * np.exp(p) / 1e6, 2),
        "forecast_low_m": np.round(now["v0"].to_numpy() * np.exp(p + lo_q) / 1e6, 2),
        "forecast_high_m": np.round(now["v0"].to_numpy() * np.exp(p + hi_q) / 1e6, 2),
        "forecast_change_pct": np.round((np.exp(p) - 1) * 100, 1),
    }).drop_duplicates("pkey")
    metrics["players_forecast"] = int(len(out))
    print(f"Market forecast: {len(out):,} players | back-test {metrics['backtest']}: "
          f"R2 of change {metrics['r2_of_change']}, miss EUR {metrics['mae_eur_m']}M "
          f"(vs {metrics['mae_eur_m_if_no_change']}M assuming no change)")
    # dated value history for the chart (players being forecast or otherwise matched)
    pk = ids.drop_duplicates("tm_player_id").set_index("tm_player_id")["pkey"]
    history = hist[hist["date"] >= "2018-01-01"].copy()
    history["pkey"] = history["player_id"].map(pk)
    history = (history.dropna(subset=["pkey"])
               .assign(value_m=lambda x: (x["market_value_in_eur"] / 1e6).round(2),
                       date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))[["pkey", "date", "value_m"]])
    return out, metrics, history
