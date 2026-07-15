# phase7/data_loader.py
"""
Loads all upstream Phase 3–6 result files.
Returns plain Python dicts — no heavy computation here.
"""

import json
import csv
import os
from phase7.config import PATHS


def _load_json(key: str) -> dict:
    path = PATHS[key]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[Phase 7] Missing upstream file: {path}\n"
            f"  → Re-run the relevant phase to generate it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(key: str) -> list:
    """Returns list of dicts (one per row)."""
    path = PATHS[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Phase 7] Missing upstream file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_all() -> dict:
    """
    Load every upstream result needed for Phase 7.
    Returns a nested dict keyed by experiment.
    """
    print("[Phase 7] Loading upstream results...")

    data = {
        "exp1": {
            "phase3":     _load_json("exp1_phase3"),
            "bayesian":   _load_json("exp1_phase6_bayesian"),
            "sequential": _load_json("exp1_phase6_sequential"),
            "novelty":    _load_json("exp1_phase6_novelty"),
            "power":      _load_json("exp1_phase6_power"),
        },
        "exp2": {
            "phase4": _load_json("exp2_phase4"),
        },
        "exp3": {
            "phase5":              _load_json("exp3_phase5"),
            "segmented_corrected": _load_csv("exp3_segmented_corrected"),
        },
    }

    print("  ✓ Experiment 1 (UPI Checkout)            — 5 files loaded")
    print("  ✓ Experiment 2 (Personalized Recs)       — 1 file loaded")
    print("  ✓ Experiment 3 (Discount Banner)          — 2 files loaded")
    return data