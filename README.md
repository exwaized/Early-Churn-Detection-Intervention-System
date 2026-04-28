# Early Churn Detection & Intervention System

Behavioral signal-based churn prediction system that detects at-risk users 7–14 days before churn occurs, assigns risk tiers, and recommends ROI-positive interventions.

## Structure

```
churn_repo/
├── data/
│   └── schema.md                  # input schema + sample
├── features/
│   ├── config.py                  # all thresholds and window sizes
│   ├── feature_engineering.py     # all feature functions + label builder
│   └── pipeline.py                # orchestrates full feature matrix build
├── analysis/
│   └── analysis.py                # hypothesis tests + KM curves + Cox PH
├── models/
│   └── modeling.py                # train (LR + XGBoost) + evaluate + SHAP
├── intervention/
│   └── intervention.py            # EV framework, tier assignment, ROI summary
├── README.md
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build feature matrix
python -m features.pipeline events.csv

# 2. Validate hypotheses + survival analysis
python -m analysis.analysis churn_features.parquet

# 3. Train + evaluate models
python -m models.modeling churn_features.parquet

# 4. Generate intervention table
python -m intervention.intervention
```

## Key Findings
- 35% activity drop vs personal baseline → 3× churn hazard (Cox HR)
- Engagement depth drops 3–5 days before frequency — earliest signal
- Top 25% high-risk users account for majority of churn events
- Optimal intervention window: within 5 days of inactivity spike

## Stack
Python · Scikit-learn · XGBoost · Lifelines · SHAP · Pandas · Matplotlib
