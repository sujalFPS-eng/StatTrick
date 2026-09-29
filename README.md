# StatTrick ⚽ | Algorithmic Scouting & Valuation Engine

StatTrick is an end-to-end MLOps football analytics platform that identifies undervalued
talent and market inefficiencies across Europe's top 5 leagues, by unifying tactical event
data (FBref) with financial transfer valuations (Transfermarkt).

> This version is a patched rebuild. See `PATCH_NOTES.md` for the full list of data,
> modelling and app fixes and why each one mattered. Numbers below are regenerated from a
> live run of the patched pipeline, not hand-typed.

## Core Architecture & Engine Capabilities

### 1. Fair-Value Modeling (`XGBoost`, out-of-fold)
A gradient-boosted regression pipeline trained on multi-season, deduplicated player
timelines to predict a player's market value from tactical output, age, position and
contract length.
* **Log-space optimisation:** targets ln(1 + value), so errors are evaluated proportionally
  rather than in flat euros.
* **Recency weighting:** derived from the calendar (not a hard-coded year), so the pipeline
  doesn't need an annual manual edit.
* **Out-of-fold predictions:** every player is scored by a model that never trained on him
  (5-fold CV), so `surplus_value_m` reflects genuine mispricing, not the model
  memorising its own training rows.
* **Conformalised quantile bands:** the 15th/85th percentile "fair value band" is calibrated
  on out-of-fold residuals so its coverage matches its nominal rate.
* **Current calibration** (regenerated `2026-09-28`): **1,971 qualified players**,
  out-of-fold **R² = 0.64 (EUR) / 0.75 (log)**, **MAE = €6.43M**, **70% band coverage**
  against a 70% nominal target. See `data/model_metrics.json` after any run.

### 2. Tactical Similarity & Cloning Index
An on-demand cosine-similarity engine (no precomputed N×N matrix - see patch notes #9) that
finds statistical clones for replacing outgoing talent or sourcing budget alternatives.
* Per-position feature weighting (passing matters more for a fullback's clone than his
  aerial rate, and so on).
* Blends shape (cosine) with magnitude (distance), so a bit-part player with the right
  *ratios* but a fraction of the volume no longer reads as a near-perfect clone.
* Every clone ships with a `Data Confidence` score: the share of the comparison that came
  from real, observed stats rather than a position-median fill-in for missing advanced data.

### 3. System Fit & Tactical Portability
Evaluates how a target's profile translates to an acquiring club's tactical identity.
* **Position-grouped club profiles:** a striker is compared to the club's attacking cohort,
  not a whole-squad average that a back four's clearances would otherwise dominate.
* **Minutes-weighted aggregation** within each position group.
* **K-Means clustering** per position group into up to four archetypes: *Positional
  Dominance & High Tilt*, *Compact Low-Block & Resiliency*, *Vertical Transition & Direct
  Counter*, *Balanced Mid-Block & Pragmatic*, matched to K-Means centroids by a greedy
  1-to-1 cosine assignment.

### 4. Automated MLOps Data Pipeline
GitHub Actions maintains a live, weekly-updating scouting database.
* **Stealth ingestion** via `seleniumbase` (Undetected Chrome) for FBref, run under a real
  virtual display rather than fully headless.
* **Fail-loud ingestion:** every stage now exits non-zero on a bad scrape or a broken
  Transfermarkt schema, so a broken run stops the job instead of silently committing a
  degraded database (see patch notes #13).
* **Health gate:** the workflow reads `data/model_metrics.json` after retraining and refuses
  to commit if the database is too small or the out-of-fold R² drops below a floor.
* **Weekly cadence:** ingest → retrain → rebuild club clusters → health gate → commit, every
  Monday 04:00 UTC.

## Technical Stack
* **Machine Learning:** `xgboost`, `scikit-learn`, `numpy`, `pandas`
* **Data Engineering:** `seleniumbase`, `rapidfuzz`, `curl_cffi`
* **Frontend:** `streamlit`, `plotly`
* **Infrastructure:** GitHub Actions (CI/CD)

## Project Structure
```text
├── .github/workflows/
│   └── mlops_pipeline.yml          # ingest -> retrain -> cluster -> health gate -> commit
├── data/
│   ├── master_scouting_db.csv      # one row per qualified player, OOF-scored
│   ├── player_timeline_db.csv      # deduplicated per-season data, NaNs intact
│   ├── club_tactical_profiles.parquet  # per-club, per-position-group K-Means clusters
│   ├── players.csv                 # Transfermarkt snapshot (refreshed weekly)
│   ├── model_metrics.json          # honest OOF R²/MAE/coverage for the current build
│   └── unmatched_players.csv       # qualified players the TM matcher couldn't place
├── src/
│   ├── common.py                   # name/club/league/season normalisation (single source)
│   ├── ingest_fbref.py             # live-season stealth scraper
│   ├── ingest_historical.py        # multi-season stealth scraper
│   ├── ingest_valuations.py        # Transfermarkt snapshot -> data/players.csv
│   ├── valuation_model.py          # dedup, aggregate, TM match, OOF XGBoost
│   ├── similarity_engine.py        # on-demand, position-weighted cosine similarity
│   └── system_fit.py               # position-grouped club clustering & fit scoring
├── frontend/
│   └── appsujal.py                 # Streamlit UI
├── requirements.txt
├── README.md
└── PATCH_NOTES.md                  # what was fixed, why, and how it was verified
```
