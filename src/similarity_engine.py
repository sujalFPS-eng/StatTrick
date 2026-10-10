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

Sample size (Oct 2026):
  - The per-90s are already time-decayed (season weights in src/common.py), but a profile built on
    1,600 minutes used to count as much as one built on 9,000. Each per-90 is now shrunk toward the
    player's position-group average in proportion to how little evidence it rests on
    (empirical Bayes): shrunk = (n * rate + k * group_rate) / (n + k), with n = recency-weighted
    90s and k estimated per stat from the data (method of moments, Poisson noise).
  - "Confidence" now combines data coverage with sample size: the position-weighted share of the
    profile that is both observed AND reliable (n / (n + k)) for both players. The old
    coverage-only number is still returned as "Data Coverage".
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

# Stats from the full event feed (FBref/Opta), which stops before the other stats do. Their
# evidence is the recency-weighted 90s times adv_minutes_share, not the whole career.
EVENT_FEED_STATS = {"prgc", "prgp", "tkl", "int", "clr", "blk", "aer_won", "psxg_net"}
# Not counts, so Poisson noise does not apply and k cannot be estimated from the data.
NON_COUNT_STATS = {"psxg_net"}
K_FALLBACK_90S = 10.0             # prior strength in 90s when k cannot be estimated (= 900 minutes)
K_BOUNDS_90S = (1.0, 60.0)


def estimate_prior_strength(rate: pd.Series, n90: pd.Series, group: pd.Series) -> float:
    """Empirical-Bayes prior strength k (in 90s) for one per-90 stat.

    Across players, var(observed rate) = var(true rate) + E[sampling noise]. For a count stat the
    noise of a per-90 built on n 90s is about group_rate / n (Poisson). The true spread tau^2 is
    what is left, and k = group_rate / tau^2 is how many 90s of a player's own data it takes before
    his rate counts as much as his position's average. Stats with a lot of real spread between
    players get a small k; noisy, low-signal stats get a large one."""
    ok = rate.notna() & (n90 > 0)
    if ok.sum() < 30:
        return K_FALLBACK_90S
    r, n, g = rate[ok], n90[ok], group[ok]
    group_rate = (r * n).groupby(g).transform("sum") / n.groupby(g).transform("sum")
    noise = group_rate.clip(lower=0) / n
    tau2 = float(((r - group_rate) ** 2 - noise).mean())
    mean_rate = float(group_rate.mean())
    if mean_rate <= 0:
        return K_FALLBACK_90S
    if not np.isfinite(tau2) or tau2 <= 0:
        return K_BOUNDS_90S[1]                     # the spread is all noise: lean on the group
    return float(np.clip(mean_rate / tau2, *K_BOUNDS_90S))


def shrink_rates(raw: pd.DataFrame, n90: pd.DataFrame, group: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Shrink every observed per-90 toward its position-group average by sample size.
    Returns (shrunk rates, reliability n / (n + k) per cell, k per column). NaN stays NaN."""
    shrunk = raw.copy()
    reliability = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns)
    ks = {}
    for col in raw.columns:
        stat = col.replace("_per90", "")
        rate, n = raw[col], n90[col]
        k = K_FALLBACK_90S if stat in NON_COUNT_STATS else estimate_prior_strength(rate, n, group)
        ks[col] = k
        ok = rate.notna() & (n > 0)
        if not ok.any():
            continue
        num = (rate.where(ok, 0.0) * n.where(ok, 0.0)).groupby(group).transform("sum")
        den = n.where(ok, 0.0).groupby(group).transform("sum")
        global_rate = float((rate[ok] * n[ok]).sum() / n[ok].sum())
        group_rate = (num / den.replace(0, np.nan)).fillna(global_rate)
        shrunk[col] = np.where(ok, (n * rate + k * group_rate) / (n + k), np.nan)
        reliability[col] = np.where(ok, n / (n + k), np.nan)
    return shrunk, reliability, ks


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
    def __init__(self, data_path=None, df: pd.DataFrame | None = None, shrink: bool = True):
        if df is None:
            from src.database import read_table          # lazy, so tests and validation can pass a frame
            df = read_table("players_master")
        self.df = df.copy()
        self.df.columns = [c.lower() for c in self.df.columns]
        self.df = self.df.dropna(subset=["player"]).drop_duplicates(subset=["player"]).reset_index(drop=True)

        self.feature_cols = [c for c in self.df.columns if c.endswith("per90")]
        self.raw = self.df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        self.observed = self.raw.notna().to_numpy()                       # True = real data, not imputed
        pos = self.df["pos_clean"] if "pos_clean" in self.df.columns else pd.Series("", index=self.df.index)
        pos = pos.fillna("").astype(str)

        # Evidence behind each per-90, in recency-weighted 90s. eff_min is written by the pipeline;
        # older tables only have career minutes, which overstate the evidence a little (no decay).
        mins_col = "eff_min" if "eff_min" in self.df.columns else "min"
        has_minutes = mins_col in self.df.columns
        n90_all = (pd.to_numeric(self.df[mins_col], errors="coerce").fillna(0.0) / 90.0 if has_minutes
                   else pd.Series(0.0, index=self.df.index))
        adv_share = (pd.to_numeric(self.df["adv_minutes_share"], errors="coerce").fillna(1.0).clip(0, 1)
                     if "adv_minutes_share" in self.df.columns else pd.Series(1.0, index=self.df.index))
        self.n90 = pd.DataFrame({c: n90_all * adv_share if c.replace("_per90", "") in EVENT_FEED_STATS else n90_all
                                 for c in self.feature_cols})

        if shrink and has_minutes:
            self.rates, rel, self.prior_strength = shrink_rates(self.raw, self.n90, pos)
            self.reliability = rel.fillna(0.0).to_numpy()
        else:                                             # no minutes to judge sample size by
            self.rates, self.prior_strength = self.raw, {}
            self.reliability = self.observed.astype(float)

        # Impute by POSITION-GROUP median (falls back to global median for tiny/empty groups),
        # so a data-less player reads as "roughly average for his position", not "does nothing".
        group_median = self.rates.groupby(pos).transform("median")
        global_median = self.rates.median()
        filled = self.rates.fillna(group_median).fillna(global_median).fillna(0.0)

        mu, sigma = filled.mean(), filled.std().replace(0, 1.0)
        self.Z = ((filled - mu) / sigma).to_numpy(dtype="float32")
        self._name_to_row = {name: i for i, name in enumerate(self.df["player"])}

    # -- internals -----------------------------------------------------------------
    def _row(self, player: str) -> int | None:
        return self._name_to_row.get(player)

    def _scored(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position-weighted similarity of row i against every other row, plus per pair:
        coverage   - share of the weighted feature mass that was OBSERVED for both players
        confidence - the same share, discounted by how reliable (sample size) each stat is for
                     both players: sum(w * sqrt(rel_i * rel_j)) / sum(w)."""
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
        coverage = (self.observed.astype(float) * obs_i * w).sum(axis=1) / max(w.sum(), 1e-6)
        confidence = (np.sqrt(self.reliability * self.reliability[i]) * w).sum(axis=1) / max(w.sum(), 1e-6)
        return score, coverage, confidence

    def explain(self, target_player: str, other_player: str, n: int = 3) -> dict | None:
        """Why are these two similar, and where do they differ? Uses only stats that were actually
        recorded for BOTH players, weighted the same way as the similarity score for the target's
        position. Returns {'alike': [...], 'differ': [...]} of (column, target value, other value)."""
        i, j = self._row(target_player), self._row(other_player)
        if i is None or j is None:
            return None
        pos = self.df.at[i, "pos_clean"] if "pos_clean" in self.df.columns else ""
        w = _weight_vector(pos, self.feature_cols)
        both = self.observed[i] & self.observed[j]
        gap = np.abs(self.Z[i] - self.Z[j])
        rows = [(k, gap[k], w[k]) for k in range(len(self.feature_cols)) if both[k]]
        if not rows:
            return {"alike": [], "differ": []}
        val = lambda k: (self.feature_cols[k], float(self.raw.iat[i, k]), float(self.raw.iat[j, k]))   # noqa: E731
        # alike: closest among the stats that matter most for this position
        key = [r for r in rows if r[2] >= 1.0] or rows
        alike = [val(k) for k, _, _ in sorted(key, key=lambda r: r[1])[:n]]
        differ = [val(k) for k, g, _ in sorted(rows, key=lambda r: -r[1] * r[2])[:n] if g >= 0.5]
        return {"alike": alike, "differ": differ}

    # -- public API ------------------------------------------------------------------
    def find_similar_players(self, target_player: str, top_n: int = 5, same_position: bool = False):
        i = self._row(target_player)
        if i is None:
            return None

        score, coverage, confidence = self._scored(i)
        score[i] = -np.inf

        mask = np.ones(len(self.df), dtype=bool)
        if same_position and "pos_clean" in self.df.columns:
            mask &= (self.df["pos_clean"] == self.df.at[i, "pos_clean"]).to_numpy()
        order = np.argsort(-np.where(mask, score, -np.inf))[:top_n]

        cols = [c for c in ["player", "squad", "pos_clean", "age_clean"] if c in self.df.columns]
        out = self.df.loc[order, cols].copy()
        out["Similarity Score"] = [f"{score[j] * 100:.1f}%" for j in order]
        out["Confidence"] = [f"{confidence[j] * 100:.0f}%" for j in order]
        out["Data Coverage"] = [f"{coverage[j] * 100:.0f}%" for j in order]
        out.rename(columns={"player": "Player", "squad": "Club", "pos_clean": "Position",
                            "age_clean": "Age"}, inplace=True)
        return out.reset_index(drop=True)


if __name__ == "__main__":
    eng = PlayerSimilarityEngine()
    sample = eng.df["player"].iloc[0]
    print(f"Smoke test - clones of {sample}:")
    print(eng.find_similar_players(sample, top_n=5))
