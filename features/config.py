# features/config.py
# Single source of truth for all thresholds and window sizes.
# Change values here — nothing else is hardcoded.
#
# CHURN_PROFILE selects the time scale of the event log:
#   "app"    : high-frequency product usage (sessions every few days)  -> short windows
#   "retail" : purchase logs (a customer buys every few weeks)         -> long windows
# Default is "retail", the profile validated on real data (UCI Online Retail II).
import os

PROFILES = {
    "app":    dict(CHURN_WINDOW_DAYS=14, SHORT_WINDOW=7,  LONG_WINDOW=14, BASELINE_WINDOW=30,
                   MIN_HISTORY_DAYS=7,  OBSERVATION_STRIDE=7),
    "retail": dict(CHURN_WINDOW_DAYS=90, SHORT_WINDOW=30, LONG_WINDOW=60, BASELINE_WINDOW=180,
                   MIN_HISTORY_DAYS=60, OBSERVATION_STRIDE=30),
}
_P = PROFILES[os.environ.get("CHURN_PROFILE", "retail")]

CHURN_WINDOW_DAYS  = _P["CHURN_WINDOW_DAYS"]   # days ahead to define churn label
SHORT_WINDOW       = _P["SHORT_WINDOW"]        # short rolling window (days)
LONG_WINDOW        = _P["LONG_WINDOW"]         # long rolling window (days)
BASELINE_WINDOW    = _P["BASELINE_WINDOW"]     # personal baseline window (days)
MIN_HISTORY_DAYS   = _P["MIN_HISTORY_DAYS"]    # drop users with less history than this
OBSERVATION_STRIDE = _P["OBSERVATION_STRIDE"]  # generate one snapshot every N days
ACTIVITY_DROP_FLAG = 0.35  # fractional drop vs personal baseline that sets flag_activity_drop
OUTLIER_CLIP_Q     = 0.99  # winsorise numeric features at this quantile
