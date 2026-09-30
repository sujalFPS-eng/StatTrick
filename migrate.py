import pandas as pd
from src.database import engine

print("Loading perfect timeline from local cache...")
df_timeline = pd.read_csv("data/player_timeline_db.csv", low_memory=False)

print("Uploading to Neon PostgreSQL...")
df_timeline.to_sql("player_timeline", con=engine, if_exists="replace", index=False)

print("Migration complete!")