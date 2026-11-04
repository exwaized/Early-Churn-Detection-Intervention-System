# models/modeling.py
"""
Time-based train/val/test split, model training, evaluation, and SHAP.

Sections:
    1. Feature columns
    2. Time-based split
    3. Training  (Logistic Regression baseline + XGBoost)
    4. Evaluation (Recall, Precision, PR-AUC, ROC-AUC, threshold tuning)
    5. SHAP explainability
    6. Risk tier assignment

Usage: python -m models.modeling churn_features.parquet
"""
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve,
)
from xgboost import XGBClassifier

from features.config import SHORT_WINDOW as SW, LONG_WINDOW as LW


# ── 1. FEATURE COLUMNS ────────────────────────────────────────────────────────

FEATURE_COLS = [
    "days_since_last_activity",
    f"sessions_{SW}d", f"sessions_{LW}d", f"active_days_{SW}d", f"active_days_{LW}d",
    "activity_slope", "activity_acceleration",
    f"session_std_{LW}d", f"session_cv_{LW}d",
    "avg_inter_event_gap", "max_inter_event_gap", "gap_trend",
    "pct_change_vs_baseline", "flag_activity_drop",
    "avg_actions_per_session", "avg_features_per_session",
    "tenure_days", "is_new_user", "is_mature_user",
]
TARGET = "churn_label"


# ── 2. TIME-BASED SPLIT ───────────────────────────────────────────────────────

def time_based_split(fm: pd.DataFrame):
    """
    Train on earliest 60%, validate on next 20%, test on final 20%.
    Never random split on temporal data — users who churn in test may have
    activity in train, leaking the label.
    """
    fm = fm.sort_values("obs_date").reset_index(drop=True)
    i1, i2 = int(len(fm) * 0.6), int(len(fm) * 0.8)
    return fm.iloc[:i1], fm.iloc[i1:i2], fm.iloc[i2:]


def prepare_xy(split: pd.DataFrame):
    sub = split[FEATURE_COLS + [TARGET]].dropna()
    return sub[FEATURE_COLS].values, sub[TARGET].values


# ── 3. TRAINING ───────────────────────────────────────────────────────────────

def train_logistic(X_train, y_train) -> Pipeline:
    """
    Baseline model. If features are solid, logistic regression already
    gives reasonable recall. Failure here = features problem, not model problem.
    """
    rate = y_train.mean()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight={0: rate, 1: 1 - rate},
                                  max_iter=500, C=0.1)),
    ])
    return pipe.fit(X_train, y_train)


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """
    Main model. scale_pos_weight handles class imbalance explicitly.
    Tune decision threshold post-training for recall/precision tradeoff.
    """
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="aucpr", random_state=42, verbosity=0,
    )
    return model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)


# ── 4. EVALUATION ─────────────────────────────────────────────────────────────

def evaluate_model(model, X, y, name: str, threshold: float = 0.5):
    """PR-AUC is the decision-relevant metric under class imbalance."""
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)
    print(f"\n── {name} (threshold={threshold}) ──")
    print(classification_report(y, pred, digits=3))
    print(f"  ROC-AUC : {roc_auc_score(y, proba):.4f}")
    print(f"  PR-AUC  : {average_precision_score(y, proba):.4f}  ← decision metric")
    return proba


def find_recall_threshold(proba, y, target_recall: float = 0.87) -> float:
    """Find lowest threshold achieving target_recall. Reports precision at that point."""
    prec, rec, thresholds = precision_recall_curve(y, proba)
    for p, r, t in zip(prec[::-1], rec[::-1], thresholds[::-1]):
        if r >= target_recall:
            print(f"  At recall={r:.3f}: precision={p:.3f}, threshold={t:.3f}")
            return float(t)
    print(f"  Could not reach recall={target_recall}")
    return 0.5


# ── 5. SHAP ───────────────────────────────────────────────────────────────────

def plot_shap(model, X, output_dir: str = "outputs"):
    """TreeSHAP — exact, not approximate. Saves bar (importance) + beeswarm (direction)."""
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    for plot_type, fname in [("bar", "shap_importance.png"), (None, "shap_beeswarm.png")]:
        plt.figure(figsize=(10, 6 if plot_type else 8))
        shap.summary_plot(shap_vals, X, feature_names=FEATURE_COLS,
                          show=False, **({"plot_type": plot_type} if plot_type else {}))
        path = f"{output_dir}/{fname}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {path}")


# ── 6. RISK TIER ──────────────────────────────────────────────────────────────

def assign_risk_tier(p_churn: float) -> str:
    if p_churn >= 0.70:
        return "HIGH"
    elif p_churn >= 0.40:
        return "MEDIUM"
    return "LOW"


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "churn_features.parquet"
    fm   = pd.read_parquet(path)

    train, val, test = time_based_split(fm)
    print(f"Split → train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")

    X_tr, y_tr = prepare_xy(train)
    X_va, y_va = prepare_xy(val)
    X_te, y_te = prepare_xy(test)

    print("\nTraining Logistic Regression baseline...")
    lr_model = train_logistic(X_tr, y_tr)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_tr, y_tr)

    print("\n=== Validation ===")
    evaluate_model(lr_model,  X_va, y_va, "Logistic Regression")
    evaluate_model(xgb_model, X_va, y_va, "XGBoost")

    print("\n=== Test (final) ===")
    proba_te = evaluate_model(xgb_model, X_te, y_te, "XGBoost")
    find_recall_threshold(proba_te, y_te)

    plot_shap(xgb_model, X_te)

    pickle.dump(lr_model,  open("outputs/lr_model.pkl",  "wb"))
    pickle.dump(xgb_model, open("outputs/xgb_model.pkl", "wb"))
    print("\nModels saved → outputs/")
