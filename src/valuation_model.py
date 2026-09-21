import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from rapidfuzz import process, fuzz


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Flattens MultiIndexes, lowercases column names, and removes duplicate headers."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "comp" in df.columns:
        df.rename(columns={"comp": "league"}, inplace=True)
    elif "competition" in df.columns:
        df.rename(columns={"competition": "league"}, inplace=True)

    return df.loc[:, ~df.columns.duplicated()].copy()


def classify_tactical_role(row: pd.Series) -> str:
    """
    Classifies players into discrete tactical positions:
    ST, WINGER, CAM, CM, CDM, FULLBACK, CB, GK
    """
    raw_pos = str(row.get("pos", "")).upper()

    if "GK" in raw_pos:
        return "GK"

    # 1. Defenders (Fullbacks vs Center-Backs)
    if "DF" in raw_pos and "FW" not in raw_pos:
        if (
            any(tag in raw_pos for tag in ["LB", "RB", "WB"])
            or row.get("prgc_per90", 0) >= 1.2
            or row.get("prgp_per90", 0) >= 2.8
        ):
            return "FULLBACK"
        return "CB"

    # 2. Attackers (Strikers vs Wingers)
    if "FW" in raw_pos:
        if any(tag in raw_pos for tag in ["LW", "RW", "LM", "RM"]) or row.get("prgc_per90", 0) >= 2.3:
            return "WINGER"
        if "MF" in raw_pos and row.get("prgp_per90", 0) >= 4.0:
            return "CAM"
        return "ST"

    # 3. Midfielders (CAM vs CM vs CDM)
    if "MF" in raw_pos:
        if (
            row.get("sh_per90", 0) >= 1.6
            or row.get("xag_per90", 0) >= 0.14
            or (row.get("gls_per90", 0) + row.get("ast_per90", 0)) >= 0.28
        ):
            return "CAM"

        defensive_actions = row.get("tkl_per90", 0) + row.get("int_per90", 0)
        if defensive_actions >= 2.8 and row.get("sh_per90", 0) < 1.2:
            return "CDM"

        return "CM"

    return "CM"

def build_valuation_model():
    print("Loading multi-season datasets...")

    # 1. Load historical seasons (2021 through 2024)
    try:
        df_history = clean_frame(pd.read_csv("data/fbref_multi_season.csv"))
    except FileNotFoundError:
        hist_files = [
            ("data/cleaned_2021-22.csv", "2021-2022"),
            ("data/cleaned_2022-23.csv", "2022-2023"),
            ("data/cleaned_2023-24.csv", "2023-2024"),
        ]
        loaded = []
        for path, season_label in hist_files:
            if os.path.exists(path):
                temp = clean_frame(pd.read_csv(path))
                if "season" not in temp.columns:
                    temp["season"] = season_label
                loaded.append(temp)
        df_history = pd.concat(loaded, ignore_index=True) if loaded else pd.DataFrame()

    # 2. Load modern seasons (2024-2025 and 2025-2026)
    try:
        df_2425 = clean_frame(pd.read_csv("data/fbref_2425.csv"))
        df_2425["season"] = "2024-2025"
    except FileNotFoundError:
        df_2425 = pd.DataFrame()
        
    try:
        df_2526 = clean_frame(pd.read_csv("data/fbref_2526.csv"))
        df_2526["season"] = "2025-2026"
    except FileNotFoundError:
        df_2526 = pd.DataFrame()

    # 3. Load freshly scraped live data via SeleniumBase
    try:
        df_live = clean_frame(pd.read_csv("data/live_fbref_data.csv"))
        df_live["season"] = "2026-2027"
    except FileNotFoundError:
        print("Live FBref data not found. Continuing with historical data only.")
        df_live = pd.DataFrame()

    # Merge master timeline
    dfs_to_concat = [d for d in [df_history, df_2425, df_2526, df_live] if not d.empty]
    df_raw = clean_frame(pd.concat(dfs_to_concat, ignore_index=True))

    if "league" not in df_raw.columns:
        df_raw["league"] = "Unknown"

    df_raw["min"] = pd.to_numeric(df_raw["min"], errors="coerce").fillna(0)

    # 4. Standardize and Map Stat Column Aliases
    alias_map = {
        "goals": "gls",
        "assists": "ast",
        "expected_goals": "xg",
        "expected_assists": "xag",
        "xa": "xag",
        "shots": "sh",
        "shots_on_target": "sot",
        "progressive_carries": "prgc",
        "prg_c": "prgc",
        "progressive_passes": "prgp",
        "prg_p": "prgp",
        "tackles": "tkl",
        "interceptions": "int",
        "clearances": "clr",
        "blocks": "blk",
        "aerials_won": "aer_won",
        "saves": "saves",
        "psxg_net": "psxg_net",
    }

    safe_rename = {k: v for k, v in alias_map.items() if k in df_raw.columns and v not in df_raw.columns}
    df_raw.rename(columns=safe_rename, inplace=True)
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()

    raw_stats = [
        "gls", "ast", "xg", "xag", "sh", "sot",
        "prgc", "prgp", "tkl", "int", "clr", "blk",
        "aer_won", "saves", "psxg_net"
    ]

    df_raw["90s"] = df_raw["min"] / 90.0

    for stat in raw_stats:
        if stat in df_raw.columns:
            col_series = df_raw[stat].iloc[:, 0] if isinstance(df_raw[stat], pd.DataFrame) else df_raw[stat]
            df_raw[stat] = pd.to_numeric(col_series, errors="coerce").fillna(0.0)
            df_raw[f"{stat}_per90"] = np.where(df_raw["90s"] > 0, df_raw[stat] / df_raw["90s"], 0.0)
        else:
            df_raw[f"{stat}_per90"] = 0.0

    stat_per90_cols = [f"{stat}_per90" for stat in raw_stats]

    # 5. Time-Decay Weighting (Shifted forward to account for the live scrape)
    season_weights = {
        "2026-2027": 1.0,
        "2025-2026": 0.85,
        "2024-2025": 0.65,
        "2023-2024": 0.45,
        "2023-24": 0.45,
        "2022-2023": 0.25,
        "2022-23": 0.25,
        "2021-2022": 0.10,
        "2021-22": 0.10,
    }

    if "season" in df_raw.columns:
        df_raw["season_weight"] = df_raw["season"].map(season_weights).fillna(0.5)
    else:
        df_raw["season_weight"] = 1.0

    df_raw["weighted_90s"] = df_raw["90s"] * df_raw["season_weight"]

    agg_dict = {
        "min": "sum",
        "weighted_90s": "sum"
    }

    for col in stat_per90_cols:
        df_raw[f"{col}_weighted"] = df_raw[col] * df_raw["weighted_90s"]
        agg_dict[f"{col}_weighted"] = "sum"
        
    # Save the un-aggregated timeline for the Form vs Baseline UI
    os.makedirs("data", exist_ok=True)
    df_raw.to_csv("data/player_timeline_db.csv", index=False)
    
    # Generate unified player profiles
    player_agg = df_raw.groupby("player", as_index=False).agg(agg_dict)

    for col in stat_per90_cols:
        player_agg[col] = np.where(
            player_agg["weighted_90s"] > 0,
            player_agg[f"{col}_weighted"] / player_agg["weighted_90s"],
            0.0
        )
        player_agg.drop(columns=[f"{col}_weighted"], inplace=True)

    player_agg.drop(columns=["weighted_90s"], inplace=True)

    # 6. Extract latest profile meta-attributes
    sort_col = "season" if "season" in df_raw.columns else "min"
    df_latest = df_raw.sort_values(sort_col, ascending=False).drop_duplicates(subset=["player"]).copy()

    meta_cols = ["player", "squad", "league", "age", "pos"]
    available_meta = [c for c in meta_cols if c in df_latest.columns]

    df_fb = df_latest[available_meta].merge(player_agg, on="player", how="left")
    df_fb = df_fb[df_fb["min"] >= 1500].copy()

    # 7. Load Transfermarkt Valuations & Contracts (Static DB in repo)
    print("Loading Transfermarkt valuation and contract data...")
    df_tm = clean_frame(pd.read_csv("data/players.csv"))

    if "name" in df_tm.columns:
        df_tm["player"] = df_tm["name"]

    if "market_value_in_eur" in df_tm.columns:
        df_tm["market_value_m"] = df_tm["market_value_in_eur"] / 1000000.0

    if "contract_expiration_date" in df_tm.columns:
        df_tm["contract_expiration_date"] = pd.to_datetime(df_tm["contract_expiration_date"], errors="coerce")
        df_tm["contract_years_left"] = (df_tm["contract_expiration_date"].dt.year - 2026).fillna(2.0).clip(lower=0, upper=7)
    else:
        df_tm["contract_years_left"] = 2.0

    tm_names = df_tm["player"].dropna().tolist()
    tm_lookup = df_tm.set_index("player")["market_value_m"].to_dict()
    tm_contract_lookup = df_tm.set_index("player")["contract_years_left"].to_dict()

    def match_player_name(fb_name):
        match = process.extractOne(fb_name, tm_names, scorer=fuzz.WRatio)
        if match and match[1] >= 85:
            return match[0]
        return None

    print("Fuzzy matching players with Transfermarkt records...")
    df_fb["tm_match_name"] = df_fb["player"].apply(match_player_name)
    df_fb["actual_value_m"] = df_fb["tm_match_name"].map(tm_lookup)
    df_fb["contract_years_left"] = df_fb["tm_match_name"].map(tm_contract_lookup).fillna(2.0)

    df_fb = df_fb.dropna(subset=["actual_value_m"]).copy()
    df_fb = df_fb[df_fb["actual_value_m"] > 0.5].copy()
    print(f"Calibrated dataset contains {len(df_fb)} qualified players.")

    # 8. Clean Age and Apply Granular Tactical Roles
    if "age" in df_fb.columns:
        df_fb["age_clean"] = df_fb["age"].astype(str).str.split("-").str[0]
        df_fb["age_clean"] = pd.to_numeric(df_fb["age_clean"], errors="coerce").fillna(25.0)
    else:
        df_fb["age_clean"] = 25.0

    print("Classifying players into granular tactical roles...")
    df_fb["pos_clean"] = df_fb.apply(classify_tactical_role, axis=1)

    # 9. Encode Features with Positional Indicators
    df_fb["raw_pos_clean"] = df_fb["pos_clean"]
    df_fb["raw_league"] = df_fb["league"]

    df_encoded = pd.get_dummies(df_fb, columns=["league", "pos_clean"], drop_first=False)

    df_encoded["pos_clean"] = df_fb["raw_pos_clean"]
    df_encoded["league"] = df_fb["raw_league"]
    df_encoded.drop(columns=["raw_pos_clean", "raw_league"], inplace=True)

    feature_candidates = [
        "age_clean", "min", "contract_years_left"
    ] + stat_per90_cols

    dummy_cols = [c for c in df_encoded.columns if c.startswith("league_") or c.startswith("pos_clean_")]
    feature_cols = [c for c in feature_candidates if c in df_encoded.columns] + dummy_cols

    X = df_encoded[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    y_raw = df_encoded["actual_value_m"].values
    y_log = np.log1p(y_raw)

    X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
        X, y_log, y_raw, test_size=0.2, random_state=42
    )

    # 10. Base Valuation Model
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train_log)

    test_predictions = np.expm1(model.predict(X_test))
    test_r2 = r2_score(y_test_raw, test_predictions)
    test_mae = mean_absolute_error(y_test_raw, test_predictions)

    print("\n--- Model Evaluation ---")
    print(f"XGBoost R² Score: {test_r2:.3f}")
    print(f"Mean Absolute Error: €{test_mae:.2f}M")

    all_predictions = np.expm1(model.predict(X))
    df_encoded["predicted_value_m"] = np.round(all_predictions, 2)

    # Quantile Regression Models (15th and 85th Percentiles for Valuation Ranges)
    try:
        model_low = XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
            quantile_alpha=0.15, random_state=42
        )
        model_low.fit(X_train, y_train_log)
        df_encoded["pred_value_low_m"] = np.round(np.expm1(model_low.predict(X)), 2)

        model_high = XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
            quantile_alpha=0.85, random_state=42
        )
        model_high.fit(X_train, y_train_log)
        df_encoded["pred_value_high_m"] = np.round(np.expm1(model_high.predict(X)), 2)
    except Exception:
        # Fallback to residual standard error if quantile objective is unavailable
        residuals = y_test_raw - test_predictions
        std_err = np.std(residuals)
        df_encoded["pred_value_low_m"] = np.round(np.clip(all_predictions - 1.04 * std_err, 0.5, None), 2)
        df_encoded["pred_value_high_m"] = np.round(all_predictions + 1.04 * std_err, 2)

    df_encoded["surplus_value_m"] = np.round(df_encoded["predicted_value_m"] - df_encoded["actual_value_m"], 2)

    os.makedirs("data", exist_ok=True)
    export_path = "data/master_scouting_db.csv"
    df_encoded.to_csv(export_path, index=False)
    print(f"Saved master scouting database to {export_path}")


if __name__ == "__main__":
    build_valuation_model()