# src/system_fit.py
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Core tactical metrics
STYLE_FEATURES = [
    "prgp_per90", "prgc_per90", "xg_per90", "xag_per90",
    "sh_per90", "tkl_per90", "int_per90", "clr_per90",
    "blk_per90", "aer_won_per90"
]

# Theoretical tactical archetype vectors (Z-score orientation)
ARCHETYPE_PROFILES = {
    "Positional Dominance & High Tilt": {
        "prgp_per90": 1.6, "prgc_per90": 1.6, "xg_per90": 1.2, "xag_per90": 1.2,
        "sh_per90": 1.0, "tkl_per90": -0.6, "int_per90": -0.6, "clr_per90": -1.4,
        "blk_per90": -1.2, "aer_won_per90": -0.4
    },
    "Compact Low-Block & Resiliency": {
        "prgp_per90": -1.4, "prgc_per90": -1.4, "xg_per90": -1.0, "xag_per90": -1.0,
        "sh_per90": -1.0, "tkl_per90": 1.0, "int_per90": 1.0, "clr_per90": 1.6,
        "blk_per90": 1.6, "aer_won_per90": 1.2
    },
    "Vertical Transition & Direct Counter": {
        "prgp_per90": -0.4, "prgc_per90": 0.8, "xg_per90": 0.6, "xag_per90": 0.2,
        "sh_per90": 1.2, "tkl_per90": 0.4, "int_per90": 0.4, "clr_per90": 0.2,
        "blk_per90": 0.0, "aer_won_per90": 0.6
    },
    "Balanced Mid-Block & Pragmatic": {
        "prgp_per90": 0.0, "prgc_per90": 0.0, "xg_per90": 0.0, "xag_per90": 0.0,
        "sh_per90": 0.0, "tkl_per90": 0.0, "int_per90": 0.0, "clr_per90": 0.0,
        "blk_per90": 0.0, "aer_won_per90": 0.0
    }
}


def match_clusters_to_archetypes(centroids: pd.DataFrame, active_features: list) -> dict:
    """
    Matches K-Means centroids to tactical archetypes using greedy cosine similarity.
    Guarantees unique 1-to-1 archetype assignment with zero label collision.
    """
    archetype_names = list(ARCHETYPE_PROFILES.keys())
    ref_vectors = []
    for name in archetype_names:
        vec = [ARCHETYPE_PROFILES[name].get(f, 0.0) for f in active_features]
        ref_vectors.append(vec)

    ref_matrix = np.array(ref_vectors)
    centroid_matrix = centroids[active_features].values

    # Compute similarity between clusters (rows) and archetypes (cols)
    sim_matrix = cosine_similarity(centroid_matrix, ref_matrix)

    matched_labels = {}
    assigned_clusters = set()
    assigned_archetypes = set()

    # Flatten coordinates and sort by highest similarity
    coords = []
    for r in range(sim_matrix.shape[0]):
        for c in range(sim_matrix.shape[1]):
            coords.append((sim_matrix[r, c], r, c))
    coords.sort(key=lambda x: x[0], reverse=True)

    # Greedy 1-to-1 matching
    for score, cluster_id, arch_idx in coords:
        if cluster_id not in assigned_clusters and arch_idx not in assigned_archetypes:
            matched_labels[cluster_id] = archetype_names[arch_idx]
            assigned_clusters.add(cluster_id)
            assigned_archetypes.add(arch_idx)
            if len(matched_labels) == 4:
                break

    return matched_labels


class SystemFitEngine:
    def __init__(
        self,
        db_path: str = "data/master_scouting_db.csv",
        club_styles_path: str = "data/club_tactical_profiles.parquet",
        force_rebuild: bool = False
    ):
        self.db_path = db_path
        self.club_styles_path = club_styles_path
        self.scaler = StandardScaler()

        if os.path.exists(self.club_styles_path) and not force_rebuild:
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

        # 1. Drop zero-variance features that aren't populated in the source scrape
        active_features = [f for f in STYLE_FEATURES if f in df.columns and df[f].std() > 1e-4]

        # 2. Filter to regular squad contributors (>= 400 mins)
        qual_players = df[df["min"] >= 400].copy()

        # 3. Drop single-player phantom squads (requires at least 7 contributors per club)
        squad_counts = qual_players.groupby("squad")["player"].count()
        valid_squads = squad_counts[squad_counts >= 7].index
        qual_players = qual_players[qual_players["squad"].isin(valid_squads)].copy()

        # 4. Compute minutes-weighted squad tactical profiles
        squad_records = []
        for (squad, league), group in qual_players.groupby(["squad", "league"]):
            total_mins = group["min"].sum()
            row = {"squad": squad, "league": league}
            for feat in active_features:
                row[feat] = (group[feat] * group["min"]).sum() / total_mins
            squad_records.append(row)

        club_agg = pd.DataFrame(squad_records)

        # 5. Scale features and fit K-Means
        scaled_matrix = self.scaler.fit_transform(club_agg[active_features])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
        club_agg["cluster_id"] = kmeans.fit_predict(scaled_matrix)

        # 6. Map archetypes using greedy profile similarity
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=active_features)
        cluster_labels = match_clusters_to_archetypes(centroids, active_features)
        club_agg["tactical_archetype"] = club_agg["cluster_id"].map(cluster_labels)

        # 7. Cache to disk
        os.makedirs(os.path.dirname(self.club_styles_path), exist_ok=True)
        club_agg.to_parquet(self.club_styles_path)
        return club_agg

    def calculate_system_fit(self, player_name: str, target_squad: str) -> dict:
        """
        Evaluates player compatibility against a club's tactical demand vector.
        """
        player_matches = self.df_players[self.df_players["player"] == player_name]
        squad_matches = self.club_profiles[self.club_profiles["squad"].str.lower() == target_squad.lower()]

        if player_matches.empty or squad_matches.empty:
            return {"fit_score": 50.0, "archetype": "Unknown", "compatibility": "Moderate"}

        player_row = player_matches.iloc[0]
        squad_row = squad_matches.iloc[0]

        # Safely extract vectors by ignoring missing columns (prevents KeyError)
        p_vec = []
        s_vec = []
        
        for feature in STYLE_FEATURES:
            p_val = player_row.get(feature, 0.0) if feature in player_row.index else 0.0
            s_val = squad_row.get(feature, 0.0) if feature in squad_row.index else 0.0
            
            p_vec.append(float(p_val) if not pd.isna(p_val) else 0.0)
            s_vec.append(float(s_val) if not pd.isna(s_val) else 0.0)
            
        p_vec = np.array(p_vec).reshape(1, -1)
        s_vec = np.array(s_vec).reshape(1, -1)

        # Normalize via Cosine Similarity
        cos_sim = cosine_similarity(p_vec, s_vec)[0][0]

        # Calibrate to 52% - 98.5% scale
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
            "archetype": squad_row.get("tactical_archetype", "Unknown"),
            "tier": tier,
            "target_squad": squad_row.get("squad", target_squad)
        }


if __name__ == "__main__":
    engine = SystemFitEngine(force_rebuild=True)
    print("✓ Club tactical clustering complete.")
    print(f"✓ Total qualified squads indexed: {len(engine.club_profiles)}")
    print("\n--- Verified Club Archetypes ---")
    sample_clubs = ["Arsenal", "Barcelona", "Manchester City", "Alavés", "Atlético Madrid", "Angers", "Everton"]
    display_df = engine.club_profiles[engine.club_profiles["squad"].isin(sample_clubs)]
    print(display_df[["squad", "league", "tactical_archetype"]].to_string(index=False))