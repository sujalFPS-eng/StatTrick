"""Club mismatches cleared by Transfermarkt's transfers file. Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from src.valuation_model import confirm_moves_from_transfers


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv("STATTRICK_TODAY", "2026-10-10")


def matched(rows):
    return pd.DataFrame(rows, columns=["player", "squad", "tm_club", "tm_player_id", "tm_club_mismatch",
                                       "contract_years_left"])


def transfers(rows):
    return pd.DataFrame(rows, columns=["player_id", "transfer_date", "from_club_name", "to_club_name"])


def test_move_to_the_club_he_plays_for_clears_the_flag():
    df = matched([["Anthony Gordon", "Barcelona", "Newcastle United", 503733, 1, 3.0]])
    tr = transfers([[503733, "2026-07-01", "Newcastle", "Barcelona"]])
    out = confirm_moves_from_transfers(df, tr).iloc[0]
    assert out["tm_club_mismatch"] == 0 and out["tm_club"] == "Barcelona" and out["tm_move_confirmed"] == 1
    assert np.isnan(out["contract_years_left"])            # the old club's contract no longer applies


def test_transfer_to_another_club_changes_nothing():
    df = matched([["Nicolo Zaniolo", "Udinese", "Galatasaray", 1, 1, np.nan]])
    tr = transfers([[1, "2026-06-30", "Atalanta", "Galatasaray"]])      # loan return, new loan missing
    out = confirm_moves_from_transfers(df, tr).iloc[0]
    assert out["tm_club_mismatch"] == 1 and out["tm_club"] == "Galatasaray"


def test_future_and_old_transfers_are_ignored():
    df = matched([["A Player", "Barcelona", "Newcastle United", 7, 1, np.nan],
                  ["B Player", "Barcelona", "Newcastle United", 8, 1, np.nan]])
    tr = transfers([[7, "2027-06-30", "Newcastle", "Barcelona"],          # not happened yet
                    [8, "2023-07-01", "Newcastle", "Barcelona"]])         # before last season
    out = confirm_moves_from_transfers(df, tr)
    assert out["tm_club_mismatch"].tolist() == [1, 1]


def test_players_without_a_mismatch_are_untouched():
    df = matched([["Bukayo Saka", "Arsenal", "Arsenal FC", 9, 0, 4.0]])
    tr = transfers([[9, "2026-07-01", "Arsenal", "Barcelona"]])
    out = confirm_moves_from_transfers(df, tr).iloc[0]
    assert out["tm_club"] == "Arsenal FC" and out["contract_years_left"] == 4.0
