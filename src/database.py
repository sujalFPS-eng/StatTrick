"""
Database configuration and schema definition.
Enforces strict typing and primary keys for the MLOps pipeline.
"""
import os
import logging
import tomllib
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, UniqueConstraint, text

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

# --- Environments -------------------------------------------------------------------------
# STATTRICK_SCHEMA unset  -> the default schema (production: what the deployed app reads)
# STATTRICK_SCHEMA=test   -> a separate Postgres schema, so a local experiment can run the whole
#                            pipeline and the app without touching production tables.
SCHEMA = os.getenv("STATTRICK_SCHEMA") or None
_STAGING_SUFFIX = "__new"


def _qualified(name: str) -> str:
    return f'"{SCHEMA}"."{name}"' if SCHEMA else f'"{name}"'


def read_table(name: str) -> pd.DataFrame:
    """Read one pipeline table from the active schema (raises ValueError if it does not exist)."""
    return pd.read_sql_table(name, con=engine, schema=SCHEMA)


def replace_tables(tables: dict) -> None:
    """Replace several tables ATOMICALLY.

    Each frame is first written in full to '<name>__new'. Only when every write has succeeded
    are the live tables dropped and the new ones renamed into place, inside ONE transaction.
    A reader therefore sees either the complete old set or the complete new set - never a
    half-written table, and never a new players_master beside an old player_timeline.
    (to_sql(if_exists="replace") dropped the live table first and refilled it row by row.)"""
    if SCHEMA and engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
    for name, df in tables.items():
        df.to_sql(name + _STAGING_SUFFIX, con=engine, schema=SCHEMA, if_exists="replace", index=False)
    with engine.begin() as conn:
        for name in tables:
            conn.execute(text(f"DROP TABLE IF EXISTS {_qualified(name)}"))
            conn.execute(text(f'ALTER TABLE {_qualified(name + _STAGING_SUFFIX)} RENAME TO "{name}"'))
    where = f"schema '{SCHEMA}'" if SCHEMA else "the default schema (production)"
    print(f"Swapped in {', '.join(tables)} -> {where}")


def init_db():
    """Creates all tables in the database if they do not exist."""
    logger.info("Initializing database schema...")
    metadata.create_all(engine)
    logger.info("Schema enforced successfully.")

if __name__ == "__main__":
    init_db()