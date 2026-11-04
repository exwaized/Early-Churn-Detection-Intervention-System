# features/pipeline.py
"""
Builds the full user-level feature matrix.
Usage : python -m features.pipeline events.csv
Output: churn_features.parquet
"""
import sys
import pandas as pd

from features.feature_engineering import (
    load_and_validate, build_observation_grid,
    recency_frequency_features, trend_features, volatility_features,
    inter_event_features, baseline_comparison_features,
    engagement_depth_features, tenure_features,
    build_churn_label, add_cohort_flags, clip_outliers, feature_churn_summary,
)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    grid        = build_observation_grid(df)
    user_groups = {uid: grp for uid, grp in df.groupby("user_id")}
    rows, total = [], len(grid)

    for i, row in grid.iterrows():
        if i % 5000 == 0:
            print(f"  {i:,} / {total:,}")
        uid, obs_date = row["user_id"], row["obs_date"]
        history = user_groups[uid]

        feat = {"user_id": uid, "obs_date": obs_date}
        feat.update(recency_frequency_features(history, obs_date))
        feat.update(trend_features(history, obs_date))
        feat.update(volatility_features(history, obs_date))
        feat.update(inter_event_features(history, obs_date))
        feat.update(baseline_comparison_features(history, obs_date))
        feat.update(engagement_depth_features(history, obs_date))
        feat.update(tenure_features(history, obs_date))
        feat["churn_label"] = build_churn_label(df, uid, obs_date)
        rows.append(feat)

    fm = add_cohort_flags(clip_outliers(pd.DataFrame(rows)))
    print(f"\nFeature matrix: {fm.shape} | Churn rate: {fm['churn_label'].mean():.1%}")
    return fm


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "events.csv"
    df   = load_and_validate(path)
    fm   = build_feature_matrix(df)
    fm.to_parquet("churn_features.parquet", index=False)
    print("Saved → churn_features.parquet\n")
    print(feature_churn_summary(fm).to_string())
