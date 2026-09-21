# src/build_matrix.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import os

def build_similarity_matrix(data_path="data/master_scouting_db.csv", output_path="data/similarity_matrix.parquet"):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Isolate numeric tactical features (ignoring metadata and financial values)
    feature_cols = [c for c in df.columns if 'per90' in c or c in ['min', 'age_clean']]
    
    # Drop rows without a player name and handle NaNs
    df = df.dropna(subset=['player']).reset_index(drop=True)
    features = df[feature_cols].fillna(0)
    
    print("Scaling features and computing Cosine Similarity...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Calculate NxN Matrix
    sim_matrix = cosine_similarity(scaled_features)
    
    # Convert to DataFrame with Player names as rows and columns
    sim_df = pd.DataFrame(sim_matrix, index=df['player'], columns=df['player'])
    
    # Save as compressed Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sim_df.to_parquet(output_path)
    print(f"Matrix successfully saved to {output_path}! Size: {sim_df.shape}")

if __name__ == "__main__":
    build_similarity_matrix()