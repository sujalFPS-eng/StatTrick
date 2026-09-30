"""
Database configuration and schema definition.
Enforces strict typing and primary keys for the MLOps pipeline.
"""
import os
import logging
import tomllib
from pathlib import Path
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, UniqueConstraint

logger = logging.getLogger(__name__)

# --- Load Database Credentials ---
db_url = None
secrets_path = Path(".streamlit/secrets.toml")

# 1. Local Run: Extract from Streamlit secrets
if secrets_path.exists():
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        try:
            db_url = secrets["connections"]["postgresql"]["url"]
        except KeyError:
            pass

# 2. GitHub Actions Run: Fallback to Environment Variables
if not db_url:
    db_url = os.getenv("DATABASE_URL")

if not db_url:
    raise ValueError("Database URL not found in secrets.toml or environment variables.")

# Neon requires sslmode=require; SQLAlchemy handles it natively
engine = create_engine(db_url, pool_pre_ping=True)
metadata = MetaData()

# --- 1. Master Scouting Database ---
players_master = Table(
    "players_master",
    metadata,
    Column("player_id", String, primary_key=True),
    Column("player", String, nullable=False),
    Column("squad", String, nullable=False),
    Column("league", String),
    Column("pos_clean", String),
    Column("age_clean", Integer),
    Column("min", Float),
    Column("contract_years_left", Float),
    Column("actual_value_m", Float),
    Column("predicted_value_m", Float),
    Column("surplus_value_m", Float),
    Column("surplus_pct", Float),
    Column("has_advanced", Boolean, default=True)
)

# --- 2. Historical Timeline ---
player_timeline = Table(
    "player_timeline",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("player", String, nullable=False),
    Column("season", String, nullable=False),
    Column("squad", String),
    Column("min", Float),
    Column("gls", Float),
    Column("ast", Float),
    Column("xg_per90", Float),
    Column("xag_per90", Float),
    Column("prgp_per90", Float),
    Column("prgc_per90", Float),
    Column("tkl_per90", Float),
    Column("int_per90", Float),
    UniqueConstraint("player", "season", name="uix_player_season")
)

# --- 3. Club Tactical Profiles ---
club_profiles = Table(
    "club_profiles",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("squad", String, nullable=False),
    Column("pos_group", String, nullable=False),
    Column("tactical_archetype", String),
    Column("prgp_per90", Float),
    Column("prgc_per90", Float),
    Column("xg_per90", Float),
    Column("xag_per90", Float),
    UniqueConstraint("squad", "pos_group", name="uix_squad_pos_group")
)

def init_db():
    """Creates all tables in the database if they do not exist."""
    logger.info("Initializing database schema...")
    metadata.create_all(engine)
    logger.info("Schema enforced successfully.")

if __name__ == "__main__":
    init_db()