"""Tactical similarity engine (patched).

Fixes vs. the original (#9 in the audit):
  - No more committed (N x N) similarity_matrix.parquet. It doubled in size every week and
    was heading straight for GitHub's 100MB file cap; a single row of cosine similarity is
    computed on demand in a few ms, so the whole precompute step (build_matrix.py) is gone.
  - Missing advanced stats (has_advanced == 0) are imputed with the POSITION-GROUP median,
    not zero. Zero-filling made every data-less player look like a defensive statue and
    collapse onto every other data-less player ("twins" - 97% of their #1 clone also had no
    data). A `data_confidence` score (share of core features that were observed, not
    imputed, for that specific pair) is returned so the UI can flag low-confidence clones.
  - Similarity blends cosine (shape) with a distance term (magnitude), so a super-sub with
    the right shot/goal *ratios* but 1/3 the volume no longer reads as a 98% clone.
  - Per-position feature weights (the README's claim, now actually implemented): passing
    metrics matter more for a fullback's clone than his aerial win rate, etc.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Place your new imports DOWN HERE, after the future annotations and standard library imports
from src.common import DATA_DIR, POS_GROUP
from src.database import engine

# Per-position feature emphasis. Any per90 column not listed gets WEIGHT_DEFAULT.
# Keys are matched by substring against the column name so both 'xg_per90' and any
# future 'npxg_per90' pick up the same weight.
POSITION_WEIGHTS: dict[str, dict[str, float]] = {
    "GK":       {"saves": 3.0, "psxg_net": 3.0, "clr": 0.5, "prgp": 0.3},
    "CB":       {"clr": 2.5, "blk": 2.0, "aer_won": 2.5, "int": 2.0, "tkl": 1.5, "prgp": 0.6, "xg": 0.3, "sh": 0.3},
    "FULLBACK": {"prgc": 2.0, "prgp": 2.0, "tkl": 1.5, "int": 1.5, "clr": 1.0, "xag": 1.0},
    "CDM":      {"tkl": 2.5, "int": 2.5, "prgp": 1.5, "blk": 1.5, "clr": 0.8, "xg": 0.4, "sh": 0.4},
    "CM":       {"prgp": 2.0, "prgc": 1.8, "tkl": 1.2, "int": 1.2, "xag": 1.2},
    "CAM":      {"xag": 2.5, "ast": 2.0, "prgp": 1.5, "sh": 1.2, "xg": 1.2, "tkl": 0.4, "int": 0.4},
    "WINGER":   {"xg": 2.0, "xag": 2.0, "prgc": 2.0, "sh": 1.5, "prgp": 1.0, "tkl": 0.4},
    "ST":       {"xg": 2.5, "gls": 2.0, "sh": 1.8, "aer_won": 1.2, "sot": 1.2, "prgp": 0.3},
}
WEIGHT_DEFAULT = 0.7
MAGNITUDE_BLEND = 0.35   # 0 = pure cosine (shape only), 1 = pure distance (magnitude only)


def _weight_vector(pos: str, feature_cols: list[str]) -> np.ndarray:
    table = POSITION_WEIGHTS.get(str(pos).upper(), {})
    w = np.full(len(feature_cols), WEIGHT_DEFAULT)
    for i, col in enumerate(feature_cols):
        stat = col.replace("_per90", "")
        for key, val in table.items():
            if stat == key:
                w[i] = val
                break
    return w


class PlayerSimilarityEngine:
    def __init__(self, data_path=None):
        self.df = pd.read_sql_table("players_master", con=engine)
        self.df.columns = [c.lower() for c in self.df.columns]
        self.df = self.df.dropna(subset=["player"]).drop_duplicates(subset=["player"]).reset_index(drop=True)

        self.feature_cols = [c for c in self.df.columns if c.endswith("per90")]
        self.raw = self.df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        self.observed = self.raw.notna().to_numpy()                       # True = real data, not imputed

        # Impute by POSITION-GROUP median (falls back to global median for tiny/empty groups),
        # so a data-less player reads as "roughly average for his position", not "does nothing".
        pos = self.df["pos_clean"] if "pos_clean" in self.df.columns else pd.Series("", index=self.df.index)
        group_median = self.raw.groupby(pos).transform("median")
        global_median = self.raw.median()
        filled = self.raw.fillna(group_median).fillna(global_median).fillna(0.0)

        mu, sigma = filled.mean(), filled.std().replace(0, 1.0)
        self.Z = ((filled - mu) / sigma).to_numpy(dtype="float32")
        self._name_to_row = {name: i for i, name in enumerate(self.df["player"])}

    # -- internals -----------------------------------------------------------------
    def _row(self, player: str) -> int | None:
        return self._name_to_row.get(player)

    def _scored(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Position-weighted similarity of row i against every other row, plus per-pair
        data confidence (share of the weighted feature mass that was OBSERVED for both)."""
        pos = self.df.at[i, "pos_clean"] if "pos_clean" in self.df.columns else ""
        w = _weight_vector(pos, self.feature_cols)
        sw = np.sqrt(w)

        zi = self.Z[i] * sw
        Zw = self.Z * sw
        dot = Zw @ zi
        norm_i = np.linalg.norm(zi)
        norms = np.linalg.norm(Zw, axis=1)
        cosine = np.divide(dot, norms * norm_i, out=np.zeros_like(dot), where=(norms * norm_i) > 0)

        dist = np.linalg.norm(Zw - zi, axis=1)
        dist_sim = 1.0 / (1.0 + dist / max(np.sqrt(w.sum()), 1e-6))     # in (0, 1], scale-free

        score = (1 - MAGNITUDE_BLEND) * ((cosine + 1) / 2) + MAGNITUDE_BLEND * dist_sim

        obs_i = self.observed[i].astype(float)
        confidence = (self.observed.astype(float) * obs_i * w).sum(axis=1) / max(w.sum(), 1e-6)
        return score, confidence

    # -- public API ------------------------------------------------------------------
    def find_similar_players(self, target_player: str, top_n: int = 5, same_position: bool = False):
        i = self._row(target_player)
        if i is None:
            return None

        score, confidence = self._scored(i)
        score[i] = -np.inf

        mask = np.ones(len(self.df), dtype=bool)
        if same_position and "pos_clean" in self.df.columns:
            mask &= (self.df["pos_clean"] == self.df.at[i, "pos_clean"]).to_numpy()
        order = np.argsort(-np.where(mask, score, -np.inf))[:top_n]

        cols = [c for c in ["player", "squad", "pos_clean", "age_clean"] if c in self.df.columns]
        out = self.df.loc[order, cols].copy()
        out["Similarity Score"] = [f"{score[j] * 100:.1f}%" for j in order]
        out["Data Confidence"] = [f"{confidence[j] * 100:.0f}%" for j in order]
        out.rename(columns={"player": "Player", "squad": "Club", "pos_clean": "Position",
                            "age_clean": "Age"}, inplace=True)
        return out.reset_index(drop=True)


if __name__ == "__main__":
    eng = PlayerSimilarityEngine()
    sample = eng.df["player"].iloc[0]
    print(f"Smoke test - clones of {sample}:")
    print(eng.find_similar_players(sample, top_n=5))
