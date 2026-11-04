# models/evaluate_real.py
"""
Evaluate the churn model on the real purchase logs and quantify the money at stake.

Everything below the "MEASURED" banner is computed from the data. The intervention
economics (save rate, cost per contact) are NOT in the data, so they are reported as a
sensitivity table with explicit assumptions, never as a result.

Inputs : churn_features.parquet (features.pipeline) and the events CSV (data/prepare_retail.py)
Usage  : python -m models.evaluate_real churn_features.parquet events.csv [results/churn_real_results.json]

Definitions
  churn_label  1 = the customer places no order in the 90 days after the snapshot date
  value        = the customer's spend in the 180 days BEFORE the snapshot (GBP). A proxy for
                 what is lost if a churner leaves; the true future value is unknowable.
  split        = time-based 60/20/20 by snapshot date; the HIGH-risk cutoff is the top decile
                 of VALIDATION scores, frozen before touching the test period.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.modeling import FEATURE_COLS, TARGET, time_based_split, train_xgboost

VALUE_WINDOW_DAYS = 180
ASSUMED_COST_PER_CONTACT_GBP = 2.0          # ASSUMPTION
ASSUMED_SAVE_RATES = [0.02, 0.05, 0.10, 0.20]  # ASSUMPTIONS, to be measured with a holdout test


def trailing_value(fm: pd.DataFrame, ev: pd.DataFrame) -> pd.Series:
    ev = ev.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    by_user = {u: g.sort_values("event_date") for u, g in ev.groupby("user_id")}
    out = []
    for uid, obs in zip(fm["user_id"], fm["obs_date"]):
        g = by_user[uid]
        m = (g["event_date"] < obs) & (g["event_date"] >= obs - pd.Timedelta(days=VALUE_WINDOW_DAYS))
        out.append(g.loc[m, "revenue"].sum())
    return pd.Series(out, index=fm.index)


def main(fm_path: str, ev_path: str, out_path: str | None):
    fm = pd.read_parquet(fm_path).reset_index(drop=True)  # keep all rows: XGBoost handles NaN natively
    ev = pd.read_csv(ev_path)
    fm["value"] = trailing_value(fm, ev)
    train, val, test = time_based_split(fm)
    X_tr, y_tr = train[FEATURE_COLS].values, train[TARGET].values
    X_va, y_va = val[FEATURE_COLS].values, val[TARGET].values
    X_te, y_te = test[FEATURE_COLS].values, test[TARGET].values

    xgb = train_xgboost(X_tr, y_tr)
    rate = y_tr.mean()
    lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                   ("lr", LogisticRegression(class_weight={0: rate, 1: 1 - rate}, max_iter=500, C=0.1))]).fit(X_tr, y_tr)
    p_va, p_te = xgb.predict_proba(X_va)[:, 1], xgb.predict_proba(X_te)[:, 1]
    p_lr = lr.predict_proba(X_te)[:, 1]
    cut = float(np.quantile(p_va, 0.90))          # HIGH-risk = top decile of validation scores

    t = test.assign(p=p_te)
    high = t[t["p"] >= cut]
    base_rate = float(y_te.mean())
    res = {
        "windows": {"churn_window_days": 90, "value_window_days": VALUE_WINDOW_DAYS},
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "test_churn_rate": round(base_rate, 4),
        "MEASURED": {
            "xgboost_roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
            "xgboost_pr_auc": round(float(average_precision_score(y_te, p_te)), 4),
            "logistic_roc_auc": round(float(roc_auc_score(y_te, p_lr)), 4),
            "logistic_pr_auc": round(float(average_precision_score(y_te, p_lr)), 4),
            "pr_auc_of_random_guessing": round(base_rate, 4),
            "top_decile_cutoff_from_validation": round(cut, 4),
            "high_risk_n": int(len(high)),
            "high_risk_churn_rate": round(float(high[TARGET].mean()), 4),
            "lift_vs_base_rate": round(float(high[TARGET].mean() / base_rate), 2),
            "share_of_all_churners_caught_in_high_risk": round(float(high[TARGET].sum() / t[TARGET].sum()), 4),
            "revenue_at_risk_gbp_high_risk_churners": round(float(high.loc[high[TARGET] == 1, "value"].sum()), 2),
            "revenue_at_risk_gbp_all_churners": round(float(t.loc[t[TARGET] == 1, "value"].sum()), 2),
        },
    }
    m = res["MEASURED"]
    m["share_of_churner_revenue_in_high_risk"] = round(
        m["revenue_at_risk_gbp_high_risk_churners"] / m["revenue_at_risk_gbp_all_churners"], 4)

    n_hi, rev_hi = len(high), m["revenue_at_risk_gbp_high_risk_churners"]
    scen = []
    for s in ASSUMED_SAVE_RATES:
        saved = rev_hi * s
        cost = n_hi * ASSUMED_COST_PER_CONTACT_GBP
        scen.append({"assumed_save_rate": s, "revenue_saved_gbp": round(saved, 2),
                     "contact_cost_gbp": round(cost, 2), "net_gbp": round(saved - cost, 2),
                     "roi_x": round(saved / cost, 2) if cost else None})
    res["ASSUMPTION_SCENARIOS"] = {
        "note": "save rates and cost are assumptions, not measured; run a holdout test to learn them",
        "assumed_cost_per_contact_gbp": ASSUMED_COST_PER_CONTACT_GBP,
        "break_even_save_rate": round(n_hi * ASSUMED_COST_PER_CONTACT_GBP / rev_hi, 4) if rev_hi else None,
        "scenarios": scen,
    }
    print(json.dumps(res, indent=2))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
