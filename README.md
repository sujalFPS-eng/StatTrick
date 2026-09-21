# StatTrick ⚽ | Algorithmic Scouting & Valuation Engine

StatTrick is an end-to-end MLOps football analytics platform engineered to identify undervalued talent and market inefficiencies across Europe's top 5 leagues. By unifying live tactical event data (FBref) with financial transfer valuations (Transfermarkt), the application provides actionable intelligence, robust statistical cloning, and system-fit modeling for professional recruitment departments.

The platform relies on a serverless CI/CD pipeline to automate stealth data ingestion, retrain machine learning valuation models, and rebuild similarity vector matrices weekly.

## Core Architecture & Engine Capabilities

### 1. Fair-Value Modeling (`XGBoost` & Quantile Regression)
A gradient-boosted regression pipeline trained on multi-season player timelines to predict a player's true market value based on tactical output, age, position, and contract length.
* **Log-Space Optimization:** Models target $\ln(1 + \text{value})$ to evaluate proportional percentage errors rather than flat monetary errors, accurately scaling from squad players to elite assets.
* **Exponential Time-Decay:** Historical seasons are down-weighted to prioritize current form while maintaining a reliable multi-year baseline.
* **Confidence Intervals:** Utilizes quantile regression (15th and 85th percentiles) to establish a "Fair Value Band" representing market variance and negotiation leverage.
* **Current Calibration:** Trained on 1,740 qualified players with **R² = 0.660** and **MAE = €7.10M**.

### 2. Tactical Similarity & Cloning Index
A vector similarity engine designed to find highly accurate statistical clones for replacing outgoing talent or sourcing budget alternatives.
* Computes cosine similarity across a $(1740, 1740)$ dimensional matrix.
* Dynamically scales per-90 tactical metrics based on granular positional importance (e.g., prioritizing progressive passing for fullbacks and ball-winning actions for defensive midfielders).

### 3. System Fit & Tactical Portability
Evaluates how seamlessly a target player's profile translates to an acquiring club's specific tactical identity.
* **Minutes-Weighted Aggregation:** Calculates squad DNA by weighting player statistics by minutes played, preventing fringe players from skewing a club's identity.
* **K-Means Clustering:** Partitions 103 active European clubs into four distinct tactical archetypes: *Positional Dominance & High Tilt*, *Compact Low-Block & Resiliency*, *Vertical Transition & Direct Counter*, and *Balanced Mid-Block & Pragmatic*.
* **Greedy Profile Matching:** Secures a 1-to-1 match between K-Means centroids and theoretical tactical vectors to prevent label collision.

### 4. Automated MLOps Data Pipeline
A serverless data engineering architecture utilizing GitHub Actions to maintain a live, constantly updating scouting database.
* **Stealth Ingestion:** Uses `seleniumbase` in Undetected Mode (UC) to bypass Cloudflare Turnstile verification and scrape live FBref weekend updates.
* **Continuous Integration:** Automatically concatenates live data to the historical database, retrains the XGBoost models, recalculates vector embeddings, and deploys artifacts back to the repository every Monday at 04:00 UTC.

## Technical Stack
* **Machine Learning & Modeling:** `xgboost`, `scikit-learn`, `numpy`, `pandas`
* **Data Engineering & Scraping:** `seleniumbase`, `rapidfuzz`
* **Frontend & Visualization:** `streamlit`, `plotly`
* **Infrastructure:** GitHub Actions (CI/CD), Streamlit Cloud

## Project Structure
```text
├── .github/workflows/
│   └── mlops_pipeline.yml              # Automated weekly ingestion & model retraining
├── data/
│   ├── master_scouting_db.csv          # Calibrated XGBoost valuation database
│   ├── player_timeline_db.csv          # Multi-season time-series data for form tracking
│   ├── similarity_matrix.parquet       # (1740x1740) Cosine similarity vector index
│   ├── club_tactical_profiles.parquet  # K-Means squad clustering artifacts
│   └── players.csv                     # Static Transfermarkt financial baseline
├── src/
│   ├── ingest_fbref.py                 # SeleniumBase stealth scraper
│   ├── valuation_model.py              # Time-decay XGBoost pipeline & quantile regression
│   ├── similarity_engine.py            # Feature scaling & matrix generation
│   └── system_fit.py                   # Minutes-weighted club clustering engine
├── frontend/
│   └── appsujal.py                     # Streamlit UI & Plotly visualization suite
├── requirements.txt                    # Production environment dependencies
└── README.md




