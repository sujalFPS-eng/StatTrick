"""Gap analysis tests. Synthetic data only: no database, no network."""
import numpy as np
import pandas as pd

from src.gap_analysis import GAP_METRICS, find_gaps, pick_gaps, rank_candidates, unit_profiles

CLUBS = [f"Club{i}" for i in range(12)]
POS = ["GK", "CB", "CB", "FULLBACK", "CM", "CDM", "CAM", "ST", "WINGER"]
ARCH = {"Low Block": {"tkl_per90": 1.0, "clr_per90": 1.6}}


def make_league(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for club in CLUBS:
        for k, pos in enumerate(POS):
            gk = pos == "GK"
            rows.append({
                "player": f"{club}_{pos}{k}", "squad": club, "pos_clean": pos,
                "min": float(rng.uniform(2000, 9000)), "age_clean": float(rng.uniform(19, 33)),
                "surplus_pct": float(rng.normal(20, 40)), "match_method": "exact",
                "edge_z": float(rng.normal(0, 1)), "outside_band": int(rng.integers(0, 2)),
                # keepers make next to no tackles: exactly the feature that used to be flagged
                "tkl_per90": 0.0 if gk else float(rng.normal(1.8, 0.4)),
                "int_per90": 0.0 if gk else float(rng.normal(1.0, 0.3)),
                "clr_per90": float(rng.normal(0.5, 0.2)) if gk else float(rng.normal(2.5, 0.8)),
                "blk_per90": 0.0 if gk else float(rng.normal(1.0, 0.3)),
                "aer_won_per90": 0.1 if gk else float(rng.normal(1.5, 0.5)),
                "prgp_per90": float(rng.normal(0.05, 0.02)) if gk else float(rng.normal(3.5, 1.0)),
                "prgc_per90": 0.0 if gk else float(rng.normal(1.5, 0.6)),
                "xg_per90": 0.0 if gk else float(rng.normal(0.2, 0.1)),
                "xag_per90": 0.0 if gk else float(rng.normal(0.15, 0.06)),
                "sh_per90": 0.0 if gk else float(rng.normal(1.5, 0.6)),
                "psxg_net_per90": float(rng.normal(0, 0.2)) if gk else np.nan,
            })
    return pd.DataFrame(rows)


def test_every_club_gets_a_goalkeeping_unit():
    units = unit_profiles(make_league())
    assert set(units.loc[units["pos_group"] == "GK", "squad"]) == set(CLUBS)


def test_goalkeepers_are_never_flagged_on_outfield_metrics():
    df = make_league()
    df.loc[df["player"] == "Club0_GK0", "psxg_net_per90"] = -1.0        # a bad shot-stopper
    gaps = find_gaps(df, "Club0")
    gk = gaps[gaps["pos_group"] == "GK"]
    assert set(gk["metric"]) <= set(GAP_METRICS["GK"])
    assert "tkl_per90" not in set(gk["metric"])
    assert gk.iloc[0]["metric"] == "psxg_net_per90" and gk.iloc[0]["gap"] > 1.0
    # every reported metric belongs to its unit's list
    for _, r in gaps.iterrows():
        assert r["metric"] in GAP_METRICS[r["pos_group"]]


def test_archetype_blueprint_sets_the_target_for_outfield_units_only():
    df = make_league()
    profiles = pd.DataFrame([{"squad": "Club0", "pos_group": "DEF", "tactical_archetype": "Low Block"},
                             {"squad": "Club0", "pos_group": "GK", "tactical_archetype": "Low Block"}])
    gaps = find_gaps(df, "Club0", club_profiles=profiles, archetype_profiles=ARCH)
    clr = gaps[(gaps["pos_group"] == "DEF") & (gaps["metric"] == "clr_per90")].iloc[0]
    assert clr["target_z"] == 1.6 and clr["archetype"] == "Low Block"
    gk = gaps[gaps["pos_group"] == "GK"].iloc[0]
    assert gk["target_z"] == 0.0 and pd.isna(gk["archetype"])


def test_all_positions_returns_one_gap_per_unit():
    df = make_league()
    # push every unit of Club0 well below average on one of its metrics
    for pos, col in [("GK", "psxg_net_per90"), ("CB", "clr_per90"), ("CM", "prgp_per90"), ("ST", "xg_per90")]:
        df.loc[(df["squad"] == "Club0") & (df["pos_clean"] == pos), col] = -5.0 if pos == "GK" else 0.0
    shown = pick_gaps(find_gaps(df, "Club0"), per_group=1, limit=4)
    assert len(shown) == 4 and set(shown["pos_group"]) == {"GK", "DEF", "MID", "ATT"}
    assert (shown["gap"] > 0).all()


def test_candidates_actually_plug_the_gap():
    df = make_league()
    out = rank_candidates(df, "tkl_per90", "MID", "Club0", max_age=40, n=50)
    assert not out.empty
    mid = df[df["pos_clean"].isin(["CM", "CDM", "CAM"])]
    assert (out["tkl_per90"] > mid["tkl_per90"].mean()).all()
    assert (out["squad"] != "Club0").all() and (out["surplus_pct"] > 0).all()
    assert out["gap_score"].is_monotonic_decreasing


def test_goalkeepers_get_extra_age_allowance():
    df = make_league()
    df.loc[df["pos_clean"] == "GK", ["age_clean", "surplus_pct", "psxg_net_per90"]] = [30.0, 50.0, 0.3]
    assert len(rank_candidates(df, "psxg_net_per90", "GK", "Club0", max_age=28, gk_age_bonus=0)) == 0
    assert len(rank_candidates(df, "psxg_net_per90", "GK", "Club0", max_age=28, n=20)) == len(CLUBS) - 1


def test_only_below_band_filter():
    df = make_league()
    out = rank_candidates(df, "xg_per90", "ATT", "Club0", max_age=40, only_below_band=True, n=50)
    assert (out["outside_band"] == 1).all()
