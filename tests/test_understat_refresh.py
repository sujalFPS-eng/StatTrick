"""Understat fallback for stale FBref rows. Synthetic data only: no database, no network."""
import numpy as np
import pandas as pd
import pytest

from src.common import make_pkey
from src.valuation_model import RAW_STATS, refresh_from_understat

CUR, PREV = "2026-2027", "2025-2026"


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("STATTRICK_TODAY", "2026-10-10")


def fbref_row(player, born, squad, season, minutes, **stats):
    row = {"player": player, "born": born, "pos": "FW", "squad": squad, "league": "Premier League",
           "age": 25, "min": minutes, "season": season, "src": "live", "rank": 1}
    row.update({s: stats.get(s, np.nan) for s in RAW_STATS})
    return row


def understat_file(tmp_path, rows):
    cols = ["player_name", "team_title", "time", "goals", "assists", "shots", "xG", "xA", "league"]
    path = tmp_path / "understat_2627.csv"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return [path]


def frame(rows):
    df = pd.DataFrame(rows)
    df["pkey"] = make_pkey(df["player"], df["born"])
    return df


def test_stale_row_takes_understat_totals_and_keeps_other_rates(tmp_path):
    df = frame([fbref_row("Bukayo Saka", 2001, "Arsenal", CUR, 450, gls=2, ast=1, sh=10, sot=4, tkl=5)])
    files = understat_file(tmp_path, [["Bukayo Saka", "Arsenal", 720, 4, 2, 18, 3.1, 1.9, "EPL"]])
    out = refresh_from_understat(df, files).iloc[0]
    assert out["min"] == 720 and out["min_src"] == "understat"
    assert (out["gls"], out["ast"], out["sh"], out["xg"], out["xag"]) == (4, 2, 18, 3.1, 1.9)
    # shots on target and tackles are not on Understat: same per-90 as FBref last measured
    assert out["sot"] / 720 == pytest.approx(4 / 450)
    assert out["tkl"] / 720 == pytest.approx(5 / 450)


def test_up_to_date_row_is_left_alone(tmp_path):
    df = frame([fbref_row("Bukayo Saka", 2001, "Arsenal", CUR, 700, gls=3, sh=15)])
    files = understat_file(tmp_path, [["Bukayo Saka", "Arsenal", 720, 4, 2, 18, 3.1, 1.9, "EPL"]])
    out = refresh_from_understat(df, files).iloc[0]
    assert out["min"] == 700 and out["gls"] == 3 and out["min_src"] == "fbref"


def test_player_missing_from_fbref_is_added_from_his_history(tmp_path):
    df = frame([fbref_row("Bukayo Saka", 2001, "Arsenal", CUR, 700),
                fbref_row("Declan Rice", 1999, "Arsenal", PREV, 3000)])
    files = understat_file(tmp_path, [["Bukayo Saka", "Arsenal", 720, 4, 2, 18, 3.1, 1.9, "EPL"],
                                      ["Declan Rice", "Arsenal", 630, 1, 2, 6, 0.8, 1.1, "EPL"]])
    out = refresh_from_understat(df, files)
    rice = out[(out["player"] == "Declan Rice") & (out["season"] == CUR)]
    assert len(rice) == 1
    rice = rice.iloc[0]
    assert rice["born"] == 1999 and rice["min"] == 630 and rice["gls"] == 1 and rice["src"] == "understat"
    assert rice["pkey"] == out.loc[out["season"] == PREV, "pkey"].iloc[0]


def test_unknown_player_without_history_is_not_added(tmp_path):
    df = frame([fbref_row("Bukayo Saka", 2001, "Arsenal", CUR, 700)])
    files = understat_file(tmp_path, [["Max Dowman", "Arsenal", 200, 1, 0, 3, 0.4, 0.2, "EPL"]])
    out = refresh_from_understat(df, files)
    assert len(out) == 1


def test_mid_season_mover_is_left_as_fbref_has_him(tmp_path):
    df = frame([fbref_row("Jadon Sancho", 2000, "Aston Villa", CUR, 300, gls=0),
                fbref_row("Jadon Sancho", 2000, "Chelsea", CUR, 200, gls=1)])
    files = understat_file(tmp_path, [["Jadon Sancho", "Aston Villa,Chelsea", 900, 2, 1, 9, 1.0, 1.0, "EPL"]])
    out = refresh_from_understat(df, files)
    assert out["min"].tolist() == [300, 200]


def test_older_seasons_are_not_touched(tmp_path):
    df = frame([fbref_row("Bukayo Saka", 2001, "Arsenal", "2023-2024", 1000, gls=5)])
    path = tmp_path / "understat_2324.csv"
    pd.DataFrame([["Bukayo Saka", "Arsenal", 2500, 14, 9, 80, 12.0, 9.0, "EPL"]],
                 columns=["player_name", "team_title", "time", "goals", "assists", "shots", "xG", "xA", "league"]
                 ).to_csv(path, index=False)
    out = refresh_from_understat(df, [path]).iloc[0]
    assert out["min"] == 1000 and out["gls"] == 5


def test_appearances_follow_understat(tmp_path):
    df = frame([fbref_row("Bukayo Saka", 2001, "Arsenal", CUR, 450, gls=2)])
    df["mp"] = 5.0
    path = tmp_path / "understat_2627.csv"
    pd.DataFrame([["Bukayo Saka", "Arsenal", 720, 8, 4, 2, 18, 3.1, 1.9, "EPL"]],
                 columns=["player_name", "team_title", "time", "games", "goals", "assists", "shots", "xG", "xA",
                          "league"]).to_csv(path, index=False)
    out = refresh_from_understat(df, [path]).iloc[0]
    assert out["mp"] == 8
