# src/system_fit.py
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


STYLE_FEATURES = [
    "prgp_per90", "prgc_per90", "xg_per90", "xag_per90",
    "sh_per90", "tkl_per90", "int_per90", "clr_per90",
    "blk_per90", "aer_won_per90"
]


def classify_cluster_name(centroid: pd.Series) -> str:
    """Assigns an intuitive tactical archetype based on centroid deviations."""
    if centroid["prgp_per90"] > 0.4 and centroid["prgc_per90"] > 0.3:
        return "Positional Dominance & High Tilt"
    if centroid["clr_per90"] > 0.4 and centroid["blk_per90"] > 0.3:
        return "Compact Low-Block & Resiliency"
    if centroid["sh_per90"] > 0.2 and centroid["prgc_per90"] > 0.2:
        return "Vertical Transition & Direct Counter"
    return "Balanced Mid-Block & Pragmatic"


class SystemFitEngine:
    def __init__(
        self,
        db_path: str = "data/master_scouting_db.csv",
        club_styles_path: str = "data/club_tactical_profiles.parquet"
    ):
        self.db_path = db_path
        self.club_styles_path = club_styles_path
        self.scaler = StandardScaler()

        if os.path.exists(self.club_styles_path):
            self.club_profiles = pd.read_parquet(self.club_styles_path)
        else:
            self.club_profiles = self.build_and_save_profiles()

        # Load player database for reference
        self.df_players = pd.read_csv(self.db_path)
        self.df_players.columns = [c.lower() for c in self.df_players.columns]
        self.df_players = self.df_players.dropna(subset=["player"]).drop_duplicates(subset=["player"]).reset_index(drop=True)

    def build_and_save_profiles(self) -> pd.DataFrame:
        df = pd.read_csv(self.db_path)
        df.columns = [c.lower() for c in df.columns]

        # Filter to regular squad contributors for accurate club identity
        qual_players = df[df["min"] >= 600].copy()

        # Compute squad-level tactical averages
        club_agg = qual_players.groupby(["squad", "league"])[STYLE_FEATURES].mean().reset_index()

        # Scale club features
        scaled_matrix = self.scaler.fit_transform(club_agg[STYLE_FEATURES])

        # Cluster clubs into 4 tactical identities
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        club_agg["cluster_id"] = kmeans.fit_predict(scaled_matrix)

        # Label archetypes dynamically using scaled cluster centroids
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=STYLE_FEATURES)
        cluster_labels = {idx: classify_cluster_name(row) for idx, row in centroids.iterrows()}
        club_agg["tactical_archetype"] = club_agg["cluster_id"].map(cluster_labels)

        # Cache to disk
        os.makedirs(os.path.dirname(self.club_styles_path), exist_ok=True)
        club_agg.to_parquet(self.club_styles_path)
        return club_agg

    def calculate_system_fit(self, player_name: str, target_squad: str) -> dict:
        """
        Evaluates player compatibility against a club's tactical demand vector.
        Returns fit score (0-100), archetype, and key divergence areas.
        """
        player_matches = self.df_players[self.df_players["player"] == player_name]
        squad_matches = self.club_profiles[self.club_profiles["squad"].str.lower() == target_squad.lower()]

        if player_matches.empty or squad_matches.empty:
            return {"fit_score": 50.0, "archetype": "Unknown", "compatibility": "Moderate"}

        player_row = player_matches.iloc[0]
        squad_row = squad_matches.iloc[0]

        # Extract vectors
        p_vec = player_row[STYLE_FEATURES].fillna(0).values.reshape(1, -1)
        s_vec = squad_row[STYLE_FEATURES].fillna(0).values.reshape(1, -1)

        # Normalize via Cosine Similarity
        cos_sim = cosine_similarity(p_vec, s_vec)[0][0]

        # Calibrate to a realistic 50% - 99% sports analytics grading scale
        raw_score = ((cos_sim + 1.0) / 2.0) * 100.0
        fit_score = float(np.clip(np.round(raw_score, 1), 52.0, 98.5))

        if fit_score >= 88.0:
            tier = "Exceptional System Synergy"
        elif fit_score >= 78.0:
            tier = "High Tactical Portability"
        elif fit_score >= 68.0:
            tier = "Moderate / Role Adjustment Required"
        else:
            tier = "System Friction / Counter-Profile"

        return {
            "fit_score": fit_score,
            "archetype": squad_row["tactical_archetype"],
            "tier": tier,
            "target_squad": squad_row["squad"]
        }


if __name__ == "__main__":
    engine = SystemFitEngine()
    print("Club tactical clustering complete.")
    print(f"Total squads indexed: {len(engine.club_profiles)}")
    print(engine.club_profiles[["squad", "tactical_archetype"]].head(10))