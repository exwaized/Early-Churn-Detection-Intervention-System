# analysis/analysis.py
"""
Statistical validation of churn hypotheses + survival analysis.

Sections:
    1. Hypothesis tests  (H1: Mann-Whitney, H2: KS-test, H3: logistic odds ratios)
    2. Activity drop threshold quantification
    3. Kaplan-Meier survival curves
    4. Cox Proportional Hazards model + Schoenfeld residual check

Usage: python -m analysis.analysis churn_features.parquet
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from features.config import LONG_WINDOW, SHORT_WINDOW


# ── CONFIG ────────────────────────────────────────────────────────────────────

DURATION_COL = "days_since_last_activity"
EVENT_COL    = "churn_label"

KM_SPLITS = {
    "cohort_activity": ["high_activity", "low_activity"],
    "cohort_tenure":   ["new", "mid", "mature"],
}

COX_FEATURES = [
    "pct_change_vs_baseline", "activity_slope", "activity_acceleration",
    "gap_trend", "avg_inter_event_gap", f"session_cv_{LONG_WINDOW}d",
    f"sessions_{SHORT_WINDOW}d", "avg_actions_per_session", "tenure_days",
]


# ── 1. HYPOTHESIS TESTS ───────────────────────────────────────────────────────

def test_h1_frequency_drop(fm: pd.DataFrame):
    """
    H1: Frequency drop precedes churn.
    Mann-Whitney U — activity data is right-skewed and zero-inflated,
    never normally distributed. MW tests systematic difference without
    assuming normality. Rank-biserial r = effect size.
    """
    print("\n── H1: Frequency drop (Mann-Whitney U) ──")
    for col in [f"sessions_{SHORT_WINDOW}d", f"sessions_{LONG_WINDOW}d", f"active_days_{SHORT_WINDOW}d"]:
        c = fm[fm[EVENT_COL] == 1][col].dropna()
        r = fm[fm[EVENT_COL] == 0][col].dropna()
        stat, p = stats.mannwhitneyu(c, r, alternative="less")
        effect  = 1 - (2 * stat) / (len(c) * len(r))
        print(f"  {col:<25} U={stat:.0f}  p={p:.4f}  effect_r={effect:.3f}")


def test_h2_inter_event(fm: pd.DataFrame):
    """
    H2: Widening inter-event gaps precede churn.
    KS test compares full distributions — mean alone misses bimodal patterns
    churners show (short gaps, then one very long gap before going dark).
    """
    print("\n── H2: Inter-event gap shift (KS test) ──")
    for col in ["avg_inter_event_gap", "max_inter_event_gap", "gap_trend"]:
        c = fm[fm[EVENT_COL] == 1][col].dropna()
        r = fm[fm[EVENT_COL] == 0][col].dropna()
        stat, p = stats.ks_2samp(c, r)
        print(f"  {col:<25} KS={stat:.3f}  p={p:.4f}")


def test_h3_volatility(fm: pd.DataFrame):
    """
    H3: Volatility predicts churn better than average activity.
    Logistic regression with just CV + activity level gives odds ratios —
    hypothesis validation, not final model.
    """
    print("\n── H3: Volatility vs activity (logistic odds ratios) ──")
    cols = [f"session_cv_{LONG_WINDOW}d", f"sessions_{LONG_WINDOW}d"]
    sub  = fm[cols + [EVENT_COL]].dropna()
    X    = StandardScaler().fit_transform(sub[cols])
    y    = sub[EVENT_COL].values
    lr   = LogisticRegression(max_iter=300).fit(X, y)
    for name, coef in zip(cols, lr.coef_[0]):
        print(f"  {name:<25} coef={coef:.3f}  odds_ratio={np.exp(coef):.3f}")


def test_activity_drop_thresholds(fm: pd.DataFrame):
    """Quantify churn rate at 20/35/50% drop thresholds. Confirms the 35% → 3x finding."""
    print("\n── Activity drop threshold analysis ──")
    for t in [0.20, 0.35, 0.50]:
        flagged  = fm[fm["pct_change_vs_baseline"] < -t]
        stable   = fm[fm["pct_change_vs_baseline"] >= -t]
        if stable.empty:
            continue
        ratio = flagged[EVENT_COL].mean() / (stable[EVENT_COL].mean() or 1)
        print(f"  >{t*100:.0f}% drop → churn {flagged[EVENT_COL].mean():.1%}  "
              f"({ratio:.1f}x vs stable, n={len(flagged):,})")


# ── 2. KAPLAN-MEIER CURVES ────────────────────────────────────────────────────

def plot_km_curves(fm: pd.DataFrame, output_dir: str = "outputs"):
    """
    KM curves per cohort segment.
    Steepest portion of each curve = your intervention window.
    Log-rank test p-value shown per split.
    """
    fig, axes = plt.subplots(1, len(KM_SPLITS), figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a0b")
    colors = ["#c8f068", "#68d4f0", "#f07068"]

    for ax, (feature, groups) in zip(axes, KM_SPLITS.items()):
        ax.set_facecolor("#111113")
        for grp, color in zip(groups, colors):
            sub = fm[fm[feature] == grp]
            if sub.empty:
                continue
            KaplanMeierFitter().fit(
                sub[DURATION_COL], sub[EVENT_COL], label=grp
            ).plot_survival_function(ax=ax, ci_show=True, color=color)

        if len(groups) >= 2:
            g1, g2 = fm[fm[feature] == groups[0]], fm[fm[feature] == groups[1]]
            if not g1.empty and not g2.empty:
                p = logrank_test(g1[DURATION_COL], g2[DURATION_COL],
                                 g1[EVENT_COL],    g2[EVENT_COL]).p_value
                ax.set_title(f"{feature}\nlog-rank p={p:.4f}",
                             color="#f0ede8", fontsize=10)

        ax.set_xlabel("Days", color="#8a8880")
        ax.set_ylabel("Survival probability", color="#8a8880")
        ax.tick_params(colors="#8a8880")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    plt.tight_layout()
    path = f"{output_dir}/km_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a0b")
    plt.close()
    print(f"Saved KM curves → {path}")


# ── 3. COX PH MODEL ───────────────────────────────────────────────────────────

def fit_cox_model(fm: pd.DataFrame) -> CoxPHFitter:
    """
    Cox PH model — hazard ratios per feature.
    Schoenfeld residuals test proportionality assumption.
    If a feature fails the test, add a time interaction for that feature.
    """
    sub = fm[[DURATION_COL, EVENT_COL] + COX_FEATURES].dropna()
    sub = sub[sub[DURATION_COL] > 0].copy()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(sub, duration_col=DURATION_COL, event_col=EVENT_COL)

    print("\n── Cox PH — Hazard Ratios ──")
    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%",
                            "exp(coef) upper 95%", "p"]].copy()
    summary.columns = ["hazard_ratio", "hr_lower_95", "hr_upper_95", "p_value"]
    print(summary.sort_values("hazard_ratio", ascending=False).round(3).to_string())

    print("\n── Proportionality check (Schoenfeld residuals) ──")
    cph.check_assumptions(sub, p_value_threshold=0.05, show_plots=False)
    return cph


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "churn_features.parquet"
    fm   = pd.read_parquet(path)
    print(f"Loaded: {fm.shape} | Churn rate: {fm[EVENT_COL].mean():.1%}")

    test_h1_frequency_drop(fm)
    test_h2_inter_event(fm)
    test_h3_volatility(fm)
    test_activity_drop_thresholds(fm)
    plot_km_curves(fm)
    fit_cox_model(fm)
