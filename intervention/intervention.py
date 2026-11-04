# intervention/intervention.py
"""
Risk-tiered intervention system using Expected Value framework +
post-processing utilities (winsorising, cohort flags, summary).

Sections:
    1. EV framework & tier assignment
    2. Intervention table builder
    3. ROI summary

Usage: python -m intervention.intervention
"""
import numpy as np
import pandas as pd


# ── 1. EV FRAMEWORK ───────────────────────────────────────────────────────────

# Adjust per business context — costs and P(save) are empirically estimated
# from a holdout control group (withheld from intervention, tracked for actual churn).
INTERVENTION_CONFIG = {
    "HIGH":   {"threshold": 0.70, "avg_cost": 12.0, "p_save": 0.35,
               "actions": ["personalised_discount", "account_manager_outreach", "feature_nudge"]},
    "MEDIUM": {"threshold": 0.40, "avg_cost": 2.0,  "p_save": 0.20,
               "actions": ["push_notification", "feature_discovery_email", "reengagement_content"]},
    "LOW":    {"threshold": 0.00, "avg_cost": 0.0,  "p_save": 0.05,
               "actions": ["standard_lifecycle_comms"]},
}


def assign_risk_tier(p_churn: float) -> str:
    if p_churn >= INTERVENTION_CONFIG["HIGH"]["threshold"]:
        return "HIGH"
    elif p_churn >= INTERVENTION_CONFIG["MEDIUM"]["threshold"]:
        return "MEDIUM"
    return "LOW"


def compute_ev(p_churn: float, ltv: float, tier: str) -> float:
    """EV = P(churn) × P(save|action) × LTV − cost. Positive EV = worthwhile."""
    cfg = INTERVENTION_CONFIG[tier]
    return round(p_churn * cfg["p_save"] * ltv - cfg["avg_cost"], 2)


# ── 2. INTERVENTION TABLE ─────────────────────────────────────────────────────

def build_intervention_table(
    user_ids: list,
    churn_probas: np.ndarray,
    ltv_per_user: np.ndarray,
) -> pd.DataFrame:
    """
    Given model outputs, return a ranked intervention table.
    Columns: user_id, churn_proba, risk_tier, recommended_actions, expected_value, intervene
    """
    records = []
    for uid, p, ltv in zip(user_ids, churn_probas, ltv_per_user):
        tier = assign_risk_tier(p)
        ev   = compute_ev(p, ltv, tier)
        records.append({
            "user_id":             uid,
            "churn_proba":         round(float(p), 4),
            "risk_tier":           tier,
            "recommended_actions": INTERVENTION_CONFIG[tier]["actions"],
            "expected_value":      ev,
            "intervene":           ev > 0,
        })
    return pd.DataFrame(records).sort_values("churn_proba", ascending=False)


# ── 3. ROI SUMMARY ────────────────────────────────────────────────────────────

def roi_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate EV, cost, and net ROI per tier."""
    rows = []
    for tier, cfg in INTERVENTION_CONFIG.items():
        sub = table[(table["risk_tier"] == tier) & (table["intervene"])]
        n   = len(sub)
        rows.append({
            "tier":       tier,
            "n_users":    n,
            "total_ev":   round(sub["expected_value"].sum(), 2),
            "total_cost": round(n * cfg["avg_cost"], 2),
            "net_roi":    round(sub["expected_value"].sum() - n * cfg["avg_cost"], 2),
        })
    return pd.DataFrame(rows)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
# NOTE: the demo below uses RANDOM probabilities and LTVs (np.random). Its output is an
# illustration of the table format, not a result. Real-data numbers: models/evaluate_real.py.

if __name__ == "__main__":
    np.random.seed(42)
    n      = 500
    uids   = [f"user_{i}" for i in range(n)]
    probas = np.random.beta(2, 5, n)
    ltvs   = np.random.normal(120, 40, n).clip(20)

    table   = build_intervention_table(uids, probas, ltvs)
    summary = roi_summary(table)

    print("=== Intervention Table (top 10) ===")
    print(table.head(10).to_string(index=False))
    print("\n=== ROI Summary by Tier ===")
    print(summary.to_string(index=False))

    table.to_csv("outputs/intervention_table.csv", index=False)
    print("\nSaved → outputs/intervention_table.csv")
