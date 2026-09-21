# src/similarity_engine.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class PlayerSimilarityEngine:
    def __init__(self, data_path="data/master_scouting_db.csv", matrix_path="data/similarity_matrix.parquet"):
        self.data_path = data_path
        self.matrix_path = matrix_path
        
        # Load dataset
        self.df = pd.read_csv(self.data_path)
        self.df.columns = [c.lower() for c in self.df.columns]
        self.df = self.df.dropna(subset=['player']).drop_duplicates(subset=['player']).reset_index(drop=True)
        
        # Keep tactical features accessible for charts & UI
        self.feature_cols = [c for c in self.df.columns if 'per90' in c or c in ['min', 'age_clean']]
        
        # Load precomputed matrix or compute fallback
        if os.path.exists(self.matrix_path):
            self.sim_matrix = pd.read_parquet(self.matrix_path)
        else:
            self._compute_matrix_fallback()

    def _compute_matrix_fallback(self):
        features = self.df[self.feature_cols].fillna(0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        sim = cosine_similarity(scaled)
        self.sim_matrix = pd.DataFrame(sim, index=self.df['player'], columns=self.df['player'])

    def find_similar_players(self, target_player: str, top_n: int = 5, same_position: bool = False):
        if target_player not in self.sim_matrix.index:
            return None

        # Extract similarity scores and remove self-comparison
        sim_scores = self.sim_matrix.loc[target_player]
        if isinstance(sim_scores, pd.DataFrame):
            sim_scores = sim_scores.iloc[0]
        sim_scores = sim_scores.drop(labels=[target_player], errors='ignore')

        # Filter by position group if requested
        if same_position and 'pos_clean' in self.df.columns:
            target_pos = self.df[self.df['player'] == target_player]['pos_clean'].iloc[0]
            valid_players = self.df[self.df['pos_clean'] == target_pos]['player']
            sim_scores = sim_scores[sim_scores.index.isin(valid_players)]

        top_matches = sim_scores.sort_values(ascending=False).head(top_n)

        # Merge with player metadata for UI display
        results = self.df[self.df['player'].isin(top_matches.index)].copy()
        results['Similarity Score'] = results['player'].map(top_matches)
        results = results.sort_values(by='Similarity Score', ascending=False)

        display_cols = ['player', 'squad', 'pos_clean', 'age_clean', 'Similarity Score']
        display_cols = [c for c in display_cols if c in results.columns]

        output = results[display_cols].copy()
        output['Similarity Score'] = (output['Similarity Score'] * 100).map('{:.1f}%'.format)
        output.rename(
            columns={
                'player': 'Player',
                'squad': 'Club',
                'pos_clean': 'Position',
                'age_clean': 'Age'
            },
            inplace=True
        )
        return output