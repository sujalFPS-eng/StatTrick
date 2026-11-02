"""Feature importances from the out-of-fold fit. Synthetic data only: no database, no network."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.valuation_model import _oof_with_contribs


def test_importances_rank_the_real_signal_first():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"age_clean": rng.uniform(18, 35, n), "min": rng.uniform(1500, 15000, n),
                      "gls_per90": rng.uniform(0, 1, n), "tkl_per90": rng.uniform(0, 3, n)})
    league = rng.choice(["A", "B", "C"], n)
    X = pd.concat([X, pd.get_dummies(pd.Series(league), prefix="league").astype(int)], axis=1)
    y = -0.15 * X["age_clean"] + 2.0 * X["gls_per90"] + 0.8 * (league == "A") + rng.normal(0, 0.1, n)

    _, _, imp = _oof_with_contribs(X, y.to_numpy(), KFold(5, shuffle=True, random_state=0), perm_repeats=2)
    feats = {r["name"]: r for r in imp["features"]}

    # dummies collapse into one 'league' variable; noise features rank last on every measure
    assert set(feats) == {"age_clean", "min", "gls_per90", "tkl_per90", "league"}
    assert feats["league"]["factor"] == "league"
    for m in ("mean_abs_shap", "perm_r2_drop", "gain_share"):
        top3 = sorted(feats, key=lambda f: -feats[f][m])[:3]
        assert set(top3) == {"age_clean", "gls_per90", "league"}, m
    assert abs(sum(r["gain_share"] for r in imp["features"]) - 1) < 1e-3
    assert {r["name"] for r in imp["factors"]} == {"age", "experience", "output", "league"}
