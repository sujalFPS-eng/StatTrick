"""Club tactical-fit engine (patched).

Fixes vs. the original (#10 in the audit):
  - Club "style" used to be the minutes-weighted average of EVERY player at the club,
    so a CB's fit score was dragged around by the club's wingers and vice versa. Profiles
    are now built PER POSITION GROUP (GK / DEF / MID / ATT) and a player is compared to his
    own group's profile, not the whole-squad blend.
  - Squad labels go through common.canon_club first, so "Newcastle" / "Newcastle United" /
    "Newcastle Utd" (previously 3 separate "clubs" in club_profiles) collapse into one.
  - K-Means runs on attacking-mid-block features common to every group; archetype matching
    is unchanged (still a 1:1 greedy match to the theoretical profiles) but is now computed
    per position group so a "high-press CB corps" can be labelled independently of the
    club's attacking identity.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.common import DATA_DIR, POS_GROUP

STYLE_FEATURES = [
    "prgp_per90", "prgc_per90", "xg_per90", "xag_per90",
    "sh_per90", "tkl_per90", "int_per90", "clr_per90",
    "blk_per90", "aer_won_per90",
]

ARCHETYPE_PROFILES = {
    "Positional Dominance & High Tilt": {
        "prgp_per90": 1.6, "prgc_per90": 1.6, "xg_per90": 1.2, "xag_per90": 1.2,
        "sh_per90": 1.0, "tkl_per90": -0.6, "int_per90": -0.6, "clr_per90": -1.4,
        "blk_per90": -1.2, "aer_won_per90": -0.4,
    },
    "Compact Low-Block & Resiliency": {
        "prgp_per90": -1.4, "prgc_per90": -1.4, "xg_per90": -1.0, "xag_per90": -1.0,
        "sh_per90": -1.0, "tkl_per90": 1.0, "int_per90": 1.0, "clr_per90": 1.6,
        "blk_per90": 1.6, "aer_won_per90": 1.2,
    },
    "Vertical Transition & Direct Counter": {
        "prgp_per90": -0.4, "prgc_per90": 0.8, "xg_per90": 0.6, "xag_per90": 0.2,
        "sh_per90": 1.2, "tkl_per90": 0.4, "int_per90": 0.4, "clr_per90": 0.2,
        "blk_per90": 0.0, "aer_won_per90": 0.6,
    },
    "Balanced Mid-Block & Pragmatic": {f: 0.0 for f in STYLE_FEATURES},
}

MIN_STINT_MINUTES = 400
MIN_GROUP_PLAYERS = 3          # a position group needs at least this many players to profile


def match_clusters_to_archetypes(centroids: pd.DataFrame, active_features: list) -> dict:
    names = list(ARCHETYPE_PROFILES.keys())
    ref = np.array([[ARCHETYPE_PROFILES[n].get(f, 0.0) for f in active_features] for n in names])
    cm = centroids.reindex(columns=active_features, fill_value=0.0).values
    sim = cosine_similarity(cm, ref)

    coords = sorted(((sim[r, c], r, c) for r in range(sim.shape[0]) for c in range(sim.shape[1])),
                    key=lambda x: -x[0])
    labels, used_c, used_a = {}, set(), set()
    for _, r, c in coords:
        if r not in used_c and c not in used_a:
            labels[r] = names[c]
            used_c.add(r); used_a.add(c)
        if len(labels) == min(sim.shape):
            break
    return labels


class SystemFitEngine:
    def __init__(self, db_path=None, club_styles_path=None, force_rebuild: bool = False):
        self.db_path = Path(db_path) if db_path else DATA_DIR / "master_scouting_db.csv"
        self.club_styles_path = Path(club_styles_path) if club_styles_path else DATA_DIR / "club_tactical_profiles.parquet"
        self.scaler = StandardScaler()

        self.df_players = pd.read_csv(self.db_path)
        self.df_players.columns = [c.lower() for c in self.df_players.columns]
        self.df_players = (self.df_players.dropna(subset=["player"])
                            .drop_duplicates(subset=["player"]).reset_index(drop=True))
        self.df_players["pos_group"] = self.df_players.get("pos_clean", "CM").map(
            lambda p: POS_GROUP.get(str(p).upper(), "MID"))

        self.feature_stats = {}
        for feat in STYLE_FEATURES:
            s = pd.to_numeric(self.df_players.get(feat), errors="coerce") if feat in self.df_players.columns else pd.Series(dtype=float)
            mean, std = s.mean(), s.std()
            self.feature_stats[feat] = (float(mean) if pd.notna(mean) else 0.0,
                                        float(std) if pd.notna(std) and std > 1e-5 else 1.0)

        if self.club_styles_path.exists() and not force_rebuild:
            self.club_profiles = pd.read_parquet(self.club_styles_path)
        else:
            self.club_profiles = self.build_and_save_profiles()

    # ------------------------------------------------------------------
    def build_and_save_profiles(self) -> pd.DataFrame:
        df = self.df_players.copy()
        active = [f for f in STYLE_FEATURES if f in df.columns and pd.to_numeric(df[f], errors="coerce").std() > 1e-4]
        qual = df[pd.to_numeric(df["min"], errors="coerce") >= MIN_STINT_MINUTES].copy()

        counts = qual.groupby(["squad", "pos_group"])["player"].transform("count")
        qual = qual[counts >= MIN_GROUP_PLAYERS].copy()

        records = []
        for (squad, league, group), g in qual.groupby(["squad", "league", "pos_group"]):
            mins = pd.to_numeric(g["min"], errors="coerce").fillna(0)
            total = mins.sum()
            if total <= 0:
                continue
            row = {"squad": squad, "league": league, "pos_group": group, "n_players": len(g)}
            for feat in active:
                vals = pd.to_numeric(g[feat], errors="coerce")
                w = mins[vals.notna()]
                v = vals.dropna()
                row[feat] = float((v * w).sum() / w.sum()) if w.sum() > 0 else np.nan
            records.append(row)

        if not records:
            return pd.DataFrame([{"squad": "Unknown", "league": "Unknown", "pos_group": "MID",
                                  "tactical_archetype": "Balanced Mid-Block & Pragmatic"}])

        club = pd.DataFrame(records).dropna(subset=active, how="all")
        out_parts = []
        for group, g in club.groupby("pos_group"):
            feats = g[active].dropna(axis=1, how="all")
            local_active = list(feats.columns)
            if len(g) < 4 or not local_active:
                g = g.copy(); g["cluster_id"] = 0
                g["tactical_archetype"] = "Balanced Mid-Block & Pragmatic"
                out_parts.append(g); continue
            X = self.scaler.fit_transform(g[local_active].fillna(g[local_active].mean()))
            k = min(4, len(g))
            km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
            g = g.copy()
            g["cluster_id"] = km.labels_
            centroids = pd.DataFrame(km.cluster_centers_, columns=local_active)
            labels = match_clusters_to_archetypes(centroids, local_active)
            g["tactical_archetype"] = g["cluster_id"].map(labels).fillna("Balanced Mid-Block & Pragmatic")
            out_parts.append(g)

        club_agg = pd.concat(out_parts, ignore_index=True)
        self.club_styles_path.parent.mkdir(parents=True, exist_ok=True)
        club_agg.to_parquet(self.club_styles_path)
        return club_agg

    # ------------------------------------------------------------------
    def calculate_system_fit(self, player_name: str, target_squad: str) -> dict:
        p = self.df_players[self.df_players["player"] == player_name]
        if p.empty:
            return {"fit_score": 50.0, "archetype": "Unknown", "tier": "Unknown", "target_squad": target_squad}
        prow = p.iloc[0]
        group = prow["pos_group"]

        s = self.club_profiles[(self.club_profiles["squad"].str.lower() == target_squad.lower())
                               & (self.club_profiles["pos_group"] == group)]
        if s.empty:                              # club has no profiled cohort in this group -> whole-squad fallback
            s = self.club_profiles[self.club_profiles["squad"].str.lower() == target_squad.lower()]
        if s.empty:
            return {"fit_score": 50.0, "archetype": "Unknown", "tier": "Unknown", "target_squad": target_squad}
        srow = s.iloc[0]

        p_vec, s_vec = [], []
        for feat in STYLE_FEATURES:
            mean, std = self.feature_stats.get(feat, (0.0, 1.0))
            pv = prow.get(feat, np.nan)
            sv = srow.get(feat, np.nan)
            p_vec.append(((float(pv) if pd.notna(pv) else mean) - mean) / std)
            s_vec.append(((float(sv) if pd.notna(sv) else mean) - mean) / std)

        p_vec, s_vec = np.array(p_vec).reshape(1, -1), np.array(s_vec).reshape(1, -1)
        cos = float(cosine_similarity(p_vec, s_vec)[0][0])
        fit_score = float(np.clip(np.round(((cos + 1) / 2) * 100, 1), 50.0, 99.0))

        tier = ("Exceptional System Synergy" if fit_score >= 85 else
                "High Tactical Portability" if fit_score >= 75 else
                "Moderate / Role Adjustment Required" if fit_score >= 65 else
                "System Friction / Counter-Profile")

        return {
            "fit_score": fit_score,
            "archetype": srow.get("tactical_archetype", "Unknown"),
            "tier": tier,
            "target_squad": srow.get("squad", target_squad),
            "position_group": group,
        }


if __name__ == "__main__":
    eng = SystemFitEngine(force_rebuild=True)
    print(f"Position-grouped club profiles built: {len(eng.club_profiles)} rows, "
          f"{eng.club_profiles['squad'].nunique()} clubs, "
          f"{eng.club_profiles.groupby('pos_group').size().to_dict()}")
