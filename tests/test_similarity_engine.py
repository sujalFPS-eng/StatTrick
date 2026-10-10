"""Unit tests for the clone engine. Synthetic data only: no database, no network."""
import numpy as np
import pandas as pd
import pytest

from src.similarity_engine import PlayerSimilarityEngine, estimate_prior_strength, shrink_rates

STATS = ["gls", "ast", "xg", "sh", "tkl", "int", "clr"]


def make_pool(n_per_pos: int = 40, seed: int = 0) -> pd.DataFrame:
    """Strikers and centre-backs with Poisson-noisy per-90s around position-specific true rates."""
    rng = np.random.default_rng(seed)
    true = {"ST": {"gls": 0.45, "ast": 0.12, "xg": 0.42, "sh": 3.0, "tkl": 0.6, "int": 0.3, "clr": 0.5},
            "CB": {"gls": 0.04, "ast": 0.03, "xg": 0.05, "sh": 0.5, "tkl": 1.5, "int": 1.4, "clr": 4.5}}
    rows = []
    for pos, rates in true.items():
        for k in range(n_per_pos):
            minutes = float(rng.uniform(1500, 12000))
            n90 = minutes / 90
            talent = rng.lognormal(0, 0.25)                      # real spread between players
            row = {"player": f"{pos}{k}", "pos_clean": pos, "min": minutes}
            for s, r in rates.items():
                row[f"{s}_per90"] = rng.poisson(r * talent * n90) / n90
            rows.append(row)
    return pd.DataFrame(rows)


def test_prior_strength_is_finite_and_bounded():
    df = make_pool()
    n90 = df["min"] / 90
    for s in STATS:
        k = estimate_prior_strength(df[f"{s}_per90"], n90, df["pos_clean"])
        assert 1.0 <= k <= 60.0


def test_small_samples_are_shrunk_more_toward_their_position():
    df = make_pool()
    # two strikers with the same extreme scoring rate, one on 1,600 minutes, one on 9,000
    extra = pd.DataFrame([
        {"player": "small", "pos_clean": "ST", "min": 1600.0, **{f"{s}_per90": 0.5 for s in STATS}, "gls_per90": 1.2},
        {"player": "large", "pos_clean": "ST", "min": 9000.0, **{f"{s}_per90": 0.5 for s in STATS}, "gls_per90": 1.2},
    ])
    df = pd.concat([df, extra], ignore_index=True)
    raw = df[[f"{s}_per90" for s in STATS]]
    n90 = pd.DataFrame({c: df["min"] / 90 for c in raw.columns})
    shrunk, rel, _ = shrink_rates(raw, n90, df["pos_clean"])
    small, large = df.index[df["player"] == "small"][0], df.index[df["player"] == "large"][0]
    assert shrunk.at[small, "gls_per90"] < shrunk.at[large, "gls_per90"] < 1.2
    assert rel.at[small, "gls_per90"] < rel.at[large, "gls_per90"]


def test_missing_values_stay_missing_after_shrinkage():
    df = make_pool()
    df.loc[0, "tkl_per90"] = np.nan
    raw = df[[f"{s}_per90" for s in STATS]]
    n90 = pd.DataFrame({c: df["min"] / 90 for c in raw.columns})
    shrunk, rel, _ = shrink_rates(raw, n90, df["pos_clean"])
    assert np.isnan(shrunk.at[0, "tkl_per90"]) and np.isnan(rel.at[0, "tkl_per90"])


def test_confidence_reflects_sample_size_coverage_does_not():
    df = make_pool()
    target = df.iloc[0].to_dict()
    twins = pd.DataFrame([{**target, "player": "twin_small", "min": 1500.0},
                          {**target, "player": "twin_large", "min": 12000.0}])
    eng = PlayerSimilarityEngine(df=pd.concat([df, twins], ignore_index=True))
    res = eng.find_similar_players(target["player"], top_n=len(eng.df)).set_index("Player")
    pct = lambda s: float(str(s).rstrip("%"))                          # noqa: E731
    assert pct(res.at["twin_small", "Confidence"]) < pct(res.at["twin_large", "Confidence"])
    assert res.at["twin_small", "Data Coverage"] == res.at["twin_large", "Data Coverage"] == "100%"


def test_imputed_stats_lower_coverage():
    df = make_pool()
    df.loc[df["player"] == "ST1", ["tkl_per90", "int_per90", "clr_per90"]] = np.nan
    eng = PlayerSimilarityEngine(df=df)
    res = eng.find_similar_players("ST0", top_n=len(eng.df)).set_index("Player")
    assert float(res.at["ST1", "Data Coverage"].rstrip("%")) < 100


def test_clones_stay_within_the_obvious_role():
    eng = PlayerSimilarityEngine(df=make_pool())
    res = eng.find_similar_players("ST0", top_n=5)
    assert (res["Position"] == "ST").all()
    assert "ST0" not in set(res["Player"])


def test_low_volume_copy_is_not_a_near_perfect_clone():
    """Same shape, a third of the output: cosine alone would call it identical."""
    df = make_pool()
    star = df[df["player"] == "ST0"].iloc[0].to_dict()
    copy = {**star, "player": "low_volume"}
    for s in STATS:
        copy[f"{s}_per90"] = star[f"{s}_per90"] / 3
    eng = PlayerSimilarityEngine(df=pd.concat([df, pd.DataFrame([copy])], ignore_index=True))
    res = eng.find_similar_players("ST0", top_n=len(eng.df)).set_index("Player")
    assert float(res.at["low_volume", "Similarity Score"].rstrip("%")) < 95


def test_shrink_off_matches_raw_rates():
    df = make_pool()
    eng = PlayerSimilarityEngine(df=df, shrink=False)
    pd.testing.assert_frame_equal(eng.rates, eng.raw)


def test_unknown_player_returns_none():
    assert PlayerSimilarityEngine(df=make_pool()).find_similar_players("nobody") is None


@pytest.mark.parametrize("col", ["min", "eff_min"])
def test_uses_recency_weighted_minutes_when_available(col):
    df = make_pool()
    if col == "eff_min":
        df["eff_min"] = df["min"] * 0.5                                 # decay halves the evidence
    eng = PlayerSimilarityEngine(df=df)
    expected = (df[col] / 90).to_numpy()
    np.testing.assert_allclose(eng.n90["gls_per90"].to_numpy(), expected)
