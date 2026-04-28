# features/config.py
# Single source of truth for all thresholds and window sizes.
# Change values here — nothing else is hardcoded.

CHURN_WINDOW_DAYS  = 14    # days ahead to define churn label
SHORT_WINDOW       = 7     # short rolling window (days)
LONG_WINDOW        = 14    # long rolling window (days)
BASELINE_WINDOW    = 30    # personal baseline window (days)
MIN_HISTORY_DAYS   = 7     # drop users with less history than this
OBSERVATION_STRIDE = 7     # generate one snapshot every N days
ACTIVITY_DROP_FLAG = 0.35  # % drop vs baseline flagging ~3x hazard
OUTLIER_CLIP_Q     = 0.99  # winsorise numeric features at this quantile
