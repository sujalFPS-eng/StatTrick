import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class PlayerSimilarityEngine:
    def __init__(self, data_path="data/master_scouting_db.csv"):
        self.df = pd.read_csv(data_path)
        self.df.columns = [c.lower() for c in self.df.columns]

        # Standardize position column if needed
        if "pos_clean" not in self.df.columns:
            pos_cols = [c for c in self.df.columns if c.startswith("pos_clean_")]
            if pos_cols:
                self.df["pos_clean"] = self.df[pos_cols].idxmax(axis=1).str.replace("pos_clean_", "")
            else:
                self.df["pos_clean"] = "MF"

        self.feature_cols = [
            "xg_per90", "xag_per90", "gls_per90", "ast_per90",
            "sh_per90", "sot_per90", "prgc_per90", "prgp_per90",
            "tkl_per90", "int_per90", "clr_per90", "blk_per90",
            "aer_won_per90", "saves_per90", "psxg_net_per90"
        ]
        self.feature_cols = [col for col in self.feature_cols if col in self.df.columns]

        for col in self.feature_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)

        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[self.feature_cols])

        # Positional Weight Matrix
        self.weight_map = {
            "FW": {"xg_per90": 2.0, "xag_per90": 1.2, "gls_per90": 2.5, "ast_per90": 1.0, "sh_per90": 1.5, "sot_per90": 1.5, "prgc_per90": 1.2, "prgp_per90": 0.8, "tkl_per90": 0.5, "int_per90": 0.5, "clr_per90": 0.2, "blk_per90": 0.2, "aer_won_per90": 1.0, "saves_per90": 0.0, "psxg_net_per90": 0.0},
            "MF": {"xg_per90": 1.0, "xag_per90": 2.0, "gls_per90": 1.0, "ast_per90": 2.0, "sh_per90": 1.0, "sot_per90": 1.0, "prgc_per90": 1.8, "prgp_per90": 2.0, "tkl_per90": 1.2, "int_per90": 1.2, "clr_per90": 0.5, "blk_per90": 0.5, "aer_won_per90": 0.5, "saves_per90": 0.0, "psxg_net_per90": 0.0},
            "DF": {"xg_per90": 0.5, "xag_per90": 0.5, "gls_per90": 0.5, "ast_per90": 0.5, "sh_per90": 0.3, "sot_per90": 0.3, "prgc_per90": 1.0, "prgp_per90": 1.5, "tkl_per90": 2.0, "int_per90": 2.0, "clr_per90": 2.5, "blk_per90": 2.5, "aer_won_per90": 2.0, "saves_per90": 0.0, "psxg_net_per90": 0.0},
            "GK": {"xg_per90": 0.0, "xag_per90": 0.0, "gls_per90": 0.0, "ast_per90": 0.0, "sh_per90": 0.0, "sot_per90": 0.0, "prgc_per90": 0.1, "prgp_per90": 0.5, "tkl_per90": 0.1, "int_per90": 0.1, "clr_per90": 1.0, "blk_per90": 0.2, "aer_won_per90": 0.5, "saves_per90": 3.0, "psxg_net_per90": 3.0}
        }

    def find_similar_players(self, player_name: str, top_n: int = 5, same_position: bool = False):
        if player_name not in self.df["player"].values:
            return f"Player '{player_name}' not found in database."

        target_idx = self.df[self.df["player"] == player_name].index[0]
        target_pos = self.df.loc[target_idx, "pos_clean"]
        
        lookup_pos = target_pos if target_pos in self.weight_map else "MF"
        weights = np.array([self.weight_map[lookup_pos].get(col, 1.0) for col in self.feature_cols])

        # Apply positional multipliers to the scaled space
        weighted_target = self.scaled_features[target_idx].reshape(1, -1) * weights
        weighted_space = self.scaled_features * weights

        similarities = cosine_similarity(weighted_target, weighted_space).flatten()
        self.df["similarity"] = similarities

        pool = self.df[self.df["player"] != player_name].copy()

        if same_position and pd.notna(target_pos):
            pool = pool[pool["pos_clean"] == target_pos]

        if pool.empty:
            return "No matching players found under the selected filters."

        top_matches = pool.sort_values(by="similarity", ascending=False).head(top_n)

        results = pd.DataFrame({
            "Player": top_matches["player"],
            "Club": top_matches["squad"].str.title() if "squad" in top_matches.columns else "N/A",
            "Pos": top_matches["pos_clean"].str.upper() if "pos_clean" in top_matches.columns else "N/A",
            "Match %": (top_matches["similarity"] * 100).round(1)
        })

        return results.reset_index(drop=True)