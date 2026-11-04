# Input Data Schema

## Required columns

| Column       | Type    | Description                     |
|--------------|---------|---------------------------------|
| `user_id`    | str/int | Unique user identifier          |
| `event_date` | date    | Session date (YYYY-MM-DD)       |
| `session_id` | str/int | Unique session identifier       |

## Optional columns

| Column           | Type | Description                              |
|------------------|------|------------------------------------------|
| `actions_count`  | int  | Actions performed in the session         |
| `features_used`  | int  | Distinct product features touched        |

## Sample

```
user_id, event_date,  session_id, actions_count, features_used
u001,    2024-01-03,  s_abc,      12,             3
u001,    2024-01-05,  s_def,      4,              1
u002,    2024-01-03,  s_xyz,      8,              2
```

One row per session. Aggregate to session level before ingestion.
Users with fewer than `MIN_HISTORY_DAYS` (see `features/config.py`) are dropped.
