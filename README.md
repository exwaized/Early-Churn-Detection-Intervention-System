# Early Churn Detection & Intervention System

Behavioral signal-based churn prediction system that scores customers for churn from behavioural signals (recency, frequency, trend, gaps, engagement depth), assigns risk tiers, and sizes the money at stake. Validated on a real purchase log (UCI Online Retail II); see *Results on real data* below.

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
pip install -r requirements.txt openpyxl

# 0. Real data: turn the UCI Online Retail II workbook into the event log
#    https://archive.ics.uci.edu/dataset/502/online+retail+ii  (CC BY 4.0)
python data/prepare_retail.py online_retail_II.xlsx events.csv

# 1. Build feature matrix  (CHURN_PROFILE=retail by default; use "app" for high-frequency product usage)
python -m features.pipeline events.csv          # ~12 min on the retail log

# 2. Validate hypotheses + survival analysis
python -m analysis.analysis churn_features.parquet

# 3. Train + evaluate, and size the money at stake
python -m models.evaluate_real churn_features.parquet events.csv results/churn_real_results.json
```

## Results on real data

Data: UCI Online Retail II, a UK online retailer, 2009-12 to 2011-12. After cleaning (no guest
checkouts, no cancellations, no non-product codes): 36,600 purchases from 5,853 customers.
A purchase is a "session". **Churn = no purchase in the next 90 days.** 42,945 customer
snapshots (one per 30 days per customer), 33.8% churn overall, 23.8% in the test period.
Time-based 60/20/20 split by snapshot date, and the HIGH-risk cutoff (top decile of validation
scores) is frozen before the test period is scored.

| Measured on the test period (8,589 snapshots) | Value |
|---|---|
| ROC-AUC, XGBoost / logistic regression | 0.683 / 0.684 |
| PR-AUC, XGBoost / logistic (random guessing = 0.238) | 0.352 / 0.368 |
| HIGH-risk group churn rate vs base rate | 36.5% vs 23.8% (1.5x lift) |
| Share of all churners inside the HIGH-risk group | 28.1% |
| Share of churner revenue inside the HIGH-risk group | 11.7% |

This is a **modest** model: better than chance, not strongly predictive. The logistic baseline
matches XGBoost, which suggests the signal is limited by the features/data, not the model.

Hypotheses (`results/analysis_output.txt`):
* A >35% drop in purchase rate vs the customer's own baseline goes with a churn rate of 36.1%,
  about **1.9x** that of customers without the drop (not the 3x previously claimed).
* Fewer purchases in the last 30/60 days, longer gaps between purchases, and higher purchase
  volatility are all significantly associated with churn (Mann-Whitney / KS / odds ratios).
* The Cox model's hazard ratios are numerically unstable on this data (some ratios are
  astronomically large), so they are not reported as findings.

### Money at stake, and what is *not* known
Measured: churners in the test period had spent GBP 801,619 in the 180 days before their snapshot;
the HIGH-risk group's churners account for GBP 94,014 of it (11.7%).

The data cannot tell us how many customers an intervention would save or what contacting them
costs, so there is **no measured ROI**. `results/churn_real_results.json` holds a sensitivity
table instead (cost assumed GBP 2 per contact, 1,573 contacts): break-even needs about a 3.4%
save rate; at an assumed 5% / 10% / 20% save rate the net is roughly GBP 1.6k / 6.3k / 15.7k.
Those save rates are assumptions. Learn the real one with a randomised holdout test.

Not previously true, now removed: the "$47 value on $12 cost (~4x ROI)" figure came from the
random demo block in `intervention/intervention.py`, not from data.

## Structure

```
.
├── data/
│   ├── schema.md                  # input schema + sample
│   └── prepare_retail.py          # UCI Online Retail II -> event log
├── features/
│   ├── config.py                  # window sizes (profiles: app, retail)
│   ├── feature_engineering.py     # feature functions + label builder
│   └── pipeline.py                # builds the full feature matrix
├── analysis/analysis.py           # hypothesis tests + KM curves + Cox PH
├── models/
│   ├── modeling.py                # LR + XGBoost, evaluation, SHAP
│   └── evaluate_real.py           # real-data evaluation + revenue at stake
├── intervention/intervention.py   # EV framework (demo block uses RANDOM inputs)
└── results/                       # measured outputs
```

## Stack
Python · Scikit-learn · XGBoost · Lifelines · SHAP · Pandas · Matplotlib
