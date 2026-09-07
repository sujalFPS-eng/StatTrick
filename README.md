# StatTrick ⚽ | Tactical Scouting & Financial Valuation Engine

StatTrick is an end-to-end football analytics platform engineered to identify undervalued talent across Europe's top leagues. By unifying tactical event data (FBref) with financial transfer valuations (Transfermarkt), the application provides actionable intelligence and robust statistical cloning for recruitment departments.

**Core Architecture**
* **Valuation Engine**: A gradient-boosted regression model (`XGBoost`) trained on multi-season timelines. It leverages exponential time-decay weighting to prioritize recent form and extracts position-specific defensive/attacking actions.
    * **Performance**: $R^2 = 0.587$ | $\text{MAE} = \text{€}6.89\text{M}$
* **Tactical Similarity Engine**: A dynamically weighted cosine similarity matrix that scales per-90 tactical metrics based on granular positional importance (e.g., heavily weighting expected goals for forwards and clearances for center-backs).
* **Frontend Interface**: A responsive, glassmorphic dark-mode dashboard built natively in Streamlit and Plotly, featuring position-grouped percentile normalization for accurate radar chart overlays and an integrated arbitrage screener.

**Project Structure**
```text
├── data/
│   └── master_scouting_db.csv      # Calibrated multi-season player database
├── src/
│   ├── valuation_model.py          # Time-decay XGBoost valuation training pipeline
│   └── similarity_engine.py        # Weighted cosine similarity clone generator
├── frontend/
│   └── appsujal.py                 # Streamlit UI & interactive visualization suite
├── requirements.txt                # Production environment dependencies
└── README.md