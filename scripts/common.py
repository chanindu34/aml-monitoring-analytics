"""
Shared configuration and helpers for the ML scripts (one source of truth).
Keeping features, model params and rules in one place prevents silent mismatches
between scripts (DRY principle).
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "aml.duckdb"
OUT_DIR = ROOT / "outputs" / "analysis"

LABEL = "label_laundering_account"

# All engineered features (see 06_account_behaviour.py)
FEATURES_ALL = [
    "txns_out", "txns_in", "out_degree", "in_degree",
    "usd_out", "usd_in", "pass_through_ratio", "median_amt_out",
    "ach_share_out", "n_formats_out", "midband_share_out", "weekend_share_out",
    "max_txns_out_per_hour", "hours_in_to_out", "active_span_hours",
]
# Final model features: active_span_hours is dropped (decision log D26).
# Ablation showed PR-AUC improves without it (0.174 to 0.181, about 3 seed std).
# Gain importance over-credited it, and it likely encodes window-timing artifacts.
FEATURES = [f for f in FEATURES_ALL if f != "active_span_hours"]

assert not any(f.startswith("label") for f in FEATURES_ALL), "Label leakage in FEATURES!"

LGBM_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    class_weight="balanced", verbose=-1,
)


def apply_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Account-level rule set (same logic as 07_rule_simulation.py)."""
    r = {
        "R1_high_in_degree":    df.in_degree >= 5,
        "R2_high_out_degree":   df.out_degree >= 5,
        "R3_ach_concentration": (df.ach_share_out.fillna(0) >= 0.5) & (df.txns_out >= 3),
        "R4_midband_volume":    (df.midband_share_out.fillna(0) >= 0.5) & (df.usd_out >= 25000),
        "R5_multi_format":      df.n_formats_out.fillna(0) >= 3,
        "R6_large_inflow":      df.usd_in >= 100000,
    }
    r["ANY_RULE_R1_to_R6"] = np.logical_or.reduce(list(r.values()))
    return {k: pd.Series(v, index=df.index).astype(bool) for k, v in r.items()}


def topk_hits(scores: np.ndarray, y: np.ndarray, k: int) -> int:
    """True positives when alerting on the k highest-scoring accounts."""
    if k <= 0:
        return 0
    idx = np.argsort(-scores, kind="stable")[:k]
    return int(y[idx].sum())


def write_csv(name: str, rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
