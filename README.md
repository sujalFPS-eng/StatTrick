# StatTrick ⚽ | Algorithmic Scouting & Valuation Engine

StatTrick is an end-to-end MLOps football analytics platform that identifies undervalued
talent and market inefficiencies across Europe's top 5 leagues, by unifying tactical event
data (FBref) with financial transfer valuations (Transfermarkt).

> This version is a patched rebuild. Each module's docstring lists the data, modelling and
> app fixes it carries and why each one mattered. Numbers below are regenerated from a
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
An on-demand similarity engine (no precomputed N×N matrix) that finds statistical clones for
replacing outgoing talent or sourcing budget alternatives.
* **Profile:** 15 time-decayed per-90s (current season 1.0, then 0.85, 0.65, 0.45, 0.25, 0.10)
  plus one role stat, recency-weighted minutes per appearance, so an impact sub and an everyday
  starter with similar per-90s are less likely to be clones.
* **Minutes shrinkage:** each per-90 is pulled toward the position average by sample size
  (empirical Bayes, prior strength estimated per stat from Poisson noise), so a 1,600-minute
  player can't look like a perfect clone by chance.
* **Score:** 80% profile shape (cosine) and 20% level (distance), with position weights at half
  strength (passing matters more for a fullback's clone than his aerial rate, and so on) and a
  light role weight.
* **Per-clone scores:** `Confidence` (share of the comparison resting on observed, well-sampled
  stats for both players), `Data Coverage` (observed stats only) and `Stability` (how often the
  clone stays in the target's top 10 across 40 resamples, each nudging every profile by about as
  much as its sample size allows).
* **Validation:** `python src/validate_clones.py` builds each player's profile from earlier
  seasons and checks whether his own test season (2024-25, 982 players) is among its closest
  matches; results go to `data/clone_validation.json` and the app's Methodology tab.

  | Engine | Own season ranked 1st | In top 10 | Median rank |
  |---|---|---|---|
  | Before Oct 2026 | 6.8% | 27.8% | 35 |
  | + minutes shrinkage | 12.1% | 36.8% | 20 |
  | Current settings | 12.5% | 40.4% | 17 |

* **Tuning:** `python src/validate_clones.py --tune` grid-searches the shape/level blend, position
  emphasis and role weight (96 combinations) on 2023-24 and reports them on 2024-25. The search
  alone would drop position weighting and the role stat (best: 13.3% / 40.0% / 17); the current
  settings keep half the position emphasis and a light role weight on purpose, because a scout
  wants someone who does the same job, not just the most identifiable statistical twin.
* **Known limits:** tactical role and team style are not modelled (per-90s are not
  possession-adjusted), so a clone from a very different side may play a different role.

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
* **FBref ingestion** via `seleniumbase` (Undetected Chrome) under a virtual display. In
  practice Cloudflare blocks it from GitHub Actions most weeks, so the step is
  `continue-on-error`: the scraper exits non-zero instead of writing a partial file, and the
  rest of the job runs on the last good FBref data.
* **Understat** (scraped weekly) backfills xG and xA (FBref lost its Opta feed after 2024-25),
  and keeps the current season fresh when FBref is blocked: a player whose Understat minutes are
  a match or more ahead of FBref takes Understat's minutes, goals, assists, shots, xG and xA, and
  a player FBref has no row for yet is added if his earlier seasons identify him. Stats Understat
  doesn't publish keep the per-90 rate FBref last measured. Progressive and defensive stats
  stop at 2024-25.
* **Fail-loud ingestion:** each stage exits non-zero on a bad scrape or a broken
  Transfermarkt schema rather than silently committing a degraded file.
* **Health gate:** the workflow reads `data/model_metrics.json` after retraining and refuses
  to commit if the database is too small or the out-of-fold R² drops below a floor.
* **Weekly cadence:** ingest → retrain → rebuild club clusters → health gate → commit, every
  Monday 04:00 UTC.
* **Tests:** `.github/workflows/tests.yml` runs `pytest` on every pull request and push to main
  (synthetic data, no database or network needed).
* **No possession adjustment:** the old "PAdj" step only rescaled columns the pipeline throws
  away, so it was removed rather than presented as adjusted data.

## Technical Stack
* **Machine Learning:** `xgboost`, `scikit-learn`, `numpy`, `pandas`
* **Data Engineering:** `seleniumbase`, `rapidfuzz`, `curl_cffi`
* **Frontend:** `streamlit`, `plotly`
* **Infrastructure:** GitHub Actions (CI/CD)

## Project Structure
```text
├── .github/workflows/
│   ├── mlops_pipeline.yml          # ingest -> retrain -> cluster -> health gate -> commit
│   └── tests.yml                   # pytest on every pull request
├── data/
│   ├── master_scouting_db.csv      # one row per qualified player, OOF-scored
│   ├── player_timeline_db.csv      # deduplicated per-season data, NaNs intact
│   ├── club_tactical_profiles.parquet  # per-club, per-position-group K-Means clusters
│   ├── players.csv                 # Transfermarkt snapshot (refreshed weekly)
│   ├── model_metrics.json          # honest OOF R²/MAE/coverage for the current build
│   ├── clone_validation.json       # out-of-sample clone check (src/validate_clones.py)
│   └── unmatched_players.csv       # qualified players the TM matcher couldn't place
├── src/
│   ├── common.py                   # name/club/league/season normalisation (single source)
│   ├── database.py                 # Postgres tables and atomic table swaps
│   ├── ingest_fbref.py             # live-season scraper (often blocked by Cloudflare in CI)
│   ├── ingest_historical.py        # multi-season scraper
│   ├── ingest_understat.py         # xG / xA backfill from Understat
│   ├── ingest_valuations.py        # Transfermarkt snapshot -> data/players.csv
│   ├── valuation_model.py          # dedup, aggregate, TM match, OOF XGBoost
│   ├── market_forecast.py          # 12-month market value forecast
│   ├── similarity_engine.py        # clone engine: shrinkage, role volume, stability
│   ├── validate_clones.py          # out-of-sample check and tuning of the clone engine
│   └── system_fit.py               # position-grouped club clustering & fit scoring
├── tests/
│   ├── test_similarity_engine.py   # clone engine (pytest, synthetic data, no database)
│   └── test_understat_refresh.py   # Understat fallback for stale FBref rows
├── frontend/
│   └── appsujal.py                 # Streamlit UI
├── requirements.txt
└── README.md
```
