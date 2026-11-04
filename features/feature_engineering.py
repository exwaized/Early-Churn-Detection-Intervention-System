# features/feature_engineering.py
"""
All feature extraction functions in one place.
Each block is self-contained and independently testable.

Sections:
    1. Data loading & validation
    2. Observation grid
    3. Feature functions (recency, trend, volatility, inter-event,
                          baseline, engagement depth, tenure)
    4. Churn label builder
    5. Post-processing (winsorise, cohort flags, summary)
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from features.config import (
    CHURN_WINDOW_DAYS, SHORT_WINDOW, LONG_WINDOW,
    BASELINE_WINDOW, MIN_HISTORY_DAYS, OBSERVATION_STRIDE,
    ACTIVITY_DROP_FLAG, OUTLIER_CLIP_Q,
)


# ── 1. LOADING & VALIDATION ──────────────────────────────────────────────────

def load_and_validate(path: str) -> pd.DataFrame:
    """
    Load raw event log and enforce schema.

    Required : user_id, event_date (YYYY-MM-DD), session_id
    Optional  : actions_count (int), features_used (int)
    """
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    missing = {"user_id", "event_date", "session_id"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    df = df.sort_values(["user_id", "event_date"]).reset_index(drop=True)

    for col in ["actions_count", "features_used"]:
        if col not in df.columns:
            df[col] = np.nan

    print(f"Loaded {len(df):,} events | {df['user_id'].nunique():,} users "
          f"| {df['event_date'].min().date()} → {df['event_date'].max().date()}")
    return df


# ── 2. OBSERVATION GRID ───────────────────────────────────────────────────────

def build_observation_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate (user_id, obs_date) pairs spaced OBSERVATION_STRIDE days apart.
    Only keeps dates with MIN_HISTORY_DAYS of prior data and
    CHURN_WINDOW_DAYS of future data available for labelling.
    """
    label_cutoff = df["event_date"].max() - pd.Timedelta(days=CHURN_WINDOW_DAYS)
    records = []

    for user_id, grp in df.groupby("user_id"):
        obs_start = grp["event_date"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS)
        obs_end   = min(grp["event_date"].max(), label_cutoff)
        if obs_start > obs_end:
            continue
        for obs_date in pd.date_range(obs_start, obs_end, freq=f"{OBSERVATION_STRIDE}D"):
            records.append({"user_id": user_id, "obs_date": obs_date})

    grid = pd.DataFrame(records)
    print(f"Observation grid: {len(grid):,} rows | {grid['user_id'].nunique():,} users")
    return grid


# ── 3. FEATURE FUNCTIONS ──────────────────────────────────────────────────────

def _daily_sessions(history: pd.DataFrame, obs_date: pd.Timestamp, window: int) -> pd.Series:
    """Sessions per day in [obs_date - window, obs_date - 1]."""
    start = obs_date - pd.Timedelta(days=window)
    mask  = (history["event_date"] >= start) & (history["event_date"] < obs_date)
    return (history[mask]
            .groupby("event_date")["session_id"]
            .nunique()
            .reindex(pd.date_range(start, obs_date - pd.Timedelta(days=1)), fill_value=0))


def recency_frequency_features(history: pd.DataFrame, obs_date: pd.Timestamp) -> dict:
    """Days since last activity, rolling session counts, active day counts."""
    past = history[history["event_date"] < obs_date]
    if past.empty:
        return {k: np.nan for k in [
            "days_since_last_activity",
            f"sessions_{SHORT_WINDOW}d", f"sessions_{LONG_WINDOW}d",
            f"active_days_{SHORT_WINDOW}d", f"active_days_{LONG_WINDOW}d",
        ]}
    feat = {"days_since_last_activity": (obs_date - past["event_date"].max()).days}
    for w in [SHORT_WINDOW, LONG_WINDOW]:
        daily = _daily_sessions(history, obs_date, w)
        feat[f"sessions_{w}d"]    = daily.sum()
        feat[f"active_days_{w}d"] = (daily > 0).sum()
    return feat


def trend_features(history: pd.DataFrame, obs_date: pd.Timestamp) -> dict:
    """
    OLS slope of daily sessions over LONG_WINDOW days.
    Acceleration = slope(second half) - slope(first half).
    Negative acceleration = decline is speeding up.
    """
    daily = _daily_sessions(history, obs_date, LONG_WINDOW).values.astype(float)
    if len(daily) < 4:
        return {"activity_slope": np.nan, "activity_acceleration": np.nan}
    try:
        x   = np.arange(len(daily))
        mid = len(daily) // 2
        slope,  *_ = linregress(x, daily)
        s1,     *_ = linregress(x[:mid], daily[:mid])
        s2,     *_ = linregress(x[mid:], daily[mid:])
        return {"activity_slope": slope, "activity_acceleration": s2 - s1}
    except Exception:
        return {"activity_slope": np.nan, "activity_acceleration": np.nan}


def volatility_features(history: pd.DataFrame, obs_date: pd.Timestamp) -> dict:
    """
    CV (std/mean) guards against false positives from erratic-but-loyal users.
    High CV = noisy baseline; go-quiet signal is less alarming.
    """
    daily = _daily_sessions(history, obs_date, LONG_WINDOW).values.astype(float)
    mean, std = daily.mean(), daily.std()
    return {
        f"session_std_{LONG_WINDOW}d": std,
        f"session_cv_{LONG_WINDOW}d":  std / mean if mean > 0 else np.nan,
    }


def inter_event_features(history: pd.DataFrame, obs_date: pd.Timestamp) -> dict:
    """
    Average gap, max gap, and slope of gap sequence.
    Widening gap trend is a pre-churn signal that appears before frequency drops.
    """
    start = obs_date - pd.Timedelta(days=LONG_WINDOW)
    mask  = (history["event_date"] >= start) & (history["event_date"] < obs_date)
    dates = (history[mask]
             .groupby("event_date")["session_id"].nunique()
             .reset_index().sort_values("event_date")["event_date"].values)

    if len(dates) < 2:
        return {"avg_inter_event_gap": np.nan, "max_inter_event_gap": np.nan, "gap_trend": np.nan}

    gaps = np.diff(dates.astype("datetime64[D]")).astype(float)
    gap_trend = linregress(np.arange(len(gaps)), gaps)[0] if len(gaps) >= 3 else np.nan
    return {
        "avg_inter_event_gap": gaps.mean(),
        "max_inter_event_gap": gaps.max(),
        "gap_trend":           gap_trend,
    }


def baseline_comparison_features(history: pd.DataFrame, obs_date: pd.Timestamp) -> dict:
    """
    % change in session rate vs user's own 30-day personal baseline.
    Normalises power users vs casual users — 40% drop means the same thing
    regardless of absolute activity level.
    """
    b_start = obs_date - pd.Timedelta(days=BASELINE_WINDOW)
    b_end   = obs_date - pd.Timedelta(days=LONG_WINDOW + 1)
    r_start = obs_date - pd.Timedelta(days=SHORT_WINDOW)

    base_rate   = (history[(history["event_date"] >= b_start) &
                           (history["event_date"] <= b_end)]["session_id"].nunique()
                   / max((b_end - b_start).days, 1))
    recent_rate = (history[(history["event_date"] >= r_start) &
                           (history["event_date"] <  obs_date)]["session_id"].nunique()
                   / SHORT_WINDOW)

    if base_rate > 0:
        pct = (recent_rate - base_rate) / base_rate
        return {"pct_change_vs_baseline": pct,
                "flag_activity_drop":     int(pct < -ACTIVITY_DROP_FLAG)}
    return {"pct_change_vs_baseline": np.nan, "flag_activity_drop": 0}


def engagement_depth_features(history: pd.DataFrame, obs_date: pd.Timestamp) -> dict:
    """
    Actions/session and features/session.
    Shallow engagement drops 3-5 days before frequency — earliest leading signal.
    """
    start = obs_date - pd.Timedelta(days=SHORT_WINDOW)
    mask  = (history["event_date"] >= start) & (history["event_date"] < obs_date)
    w     = history[mask]

    if w.empty or w["actions_count"].isna().all():
        return {"avg_actions_per_session": np.nan, "avg_features_per_session": np.nan}

    agg = w.groupby("session_id").agg(
        actions=("actions_count",  "sum"),
        features=("features_used", "mean"),
    )
    return {"avg_actions_per_session":  agg["actions"].mean(),
            "avg_features_per_session": agg["features"].mean()}


def tenure_features(history: pd.DataFrame, obs_date: pd.Timestamp) -> dict:
    """User age at observation date. Always include — churn dynamics differ by tenure."""
    days = (obs_date - history["event_date"].min()).days
    return {"tenure_days": days, "is_new_user": int(days < 30), "is_mature_user": int(days >= 90)}


# ── 4. CHURN LABEL ────────────────────────────────────────────────────────────

def build_churn_label(df: pd.DataFrame, user_id, obs_date: pd.Timestamp) -> int:
    """1 = zero activity in next CHURN_WINDOW_DAYS, 0 = retained."""
    mask = (
        (df["user_id"]    == user_id) &
        (df["event_date"] >  obs_date) &
        (df["event_date"] <= obs_date + pd.Timedelta(days=CHURN_WINDOW_DAYS))
    )
    return int(df[mask].empty)


# ── 5. POST-PROCESSING ────────────────────────────────────────────────────────

def add_cohort_flags(fm: pd.DataFrame) -> pd.DataFrame:
    """Cohort segmentation for stratified analysis. Not used as model features."""
    median = fm[f"sessions_{LONG_WINDOW}d"].median()
    fm["cohort_activity"] = np.where(fm[f"sessions_{LONG_WINDOW}d"] >= median,
                                     "high_activity", "low_activity")
    fm["cohort_tenure"]   = np.where(fm["is_new_user"] == 1, "new",
                             np.where(fm["is_mature_user"] == 1, "mature", "mid"))
    return fm


def clip_outliers(fm: pd.DataFrame) -> pd.DataFrame:
    """Winsorise at OUTLIER_CLIP_Q. Logistic regression needs this; tree models tolerate it."""
    exclude = {"churn_label", "is_new_user", "is_mature_user", "flag_activity_drop"}
    for col in [c for c in fm.select_dtypes(include=[np.number]).columns if c not in exclude]:
        fm[col] = fm[col].clip(upper=fm[col].quantile(OUTLIER_CLIP_Q))
    return fm


def feature_churn_summary(fm: pd.DataFrame) -> pd.DataFrame:
    """Mean per feature split by churned vs retained. Run before modeling as a sanity check."""
    cols     = [c for c in fm.select_dtypes(include=[np.number]).columns if c != "churn_label"]
    churned  = fm[fm["churn_label"] == 1][cols].mean()
    retained = fm[fm["churn_label"] == 0][cols].mean()
    diff_pct = ((churned - retained) / retained.replace(0, np.nan) * 100).round(1)
    return pd.DataFrame({"retained_mean": retained.round(3),
                         "churned_mean":  churned.round(3),
                         "pct_diff":      diff_pct}).sort_values("pct_diff")
