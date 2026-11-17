"""Synthetic check of features/external/harness.py (IDSSE tuning harness; data: Bassek, Rein,
Weber et al., Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y, CC BY 4.0). Tiny made-up
tables via an injected loader; no real files.

    .venv/bin/python features/tests/test_harness.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external"))
import harness as hz  # noqa: E402


def table(labels_p1, labels_p2):
    lab = list(labels_p1) + list(labels_p2)
    n = len(lab)
    return pd.DataFrame({"half": np.array([1] * len(labels_p1) + [2] * len(labels_p2), np.int8),
                         "label": pd.Categorical(lab, categories=["H", "A", "D"]),
                         "ball_x_m": np.zeros(n, np.float32),
                         "nearest_team": np.zeros(n, np.int8),
                         "ball_dropped": np.zeros(n, bool)})


def raises(exc, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc:
        return
    raise AssertionError("{} did not raise {}".format(fn.__name__, exc.__name__))


def main():
    # Seam: H at the end of period 1 and the start of period 2 must be two GT spells, not one.
    df = table("DDHHHH", "HHHHDD")
    res = hz.score_match(df, np.asarray(df["label"]).astype(str))
    assert res["n_gt_spells"] == 2, res["n_gt_spells"]
    assert res["n_frames"] == len(df) + hz.SEAM_ROWS and res["n_graded"] == len(df), "seam adds X only"
    assert res["spell_f1_mean"] == 1.0, res["spell_f1_mean"]  # (macro_f1 < 1 here: no A class)
    raises(AssertionError, hz.with_seam, np.zeros(4, np.int8), np.array([1, 2, 1, 2]))
    raises(AssertionError, hz.with_seam, np.zeros(4, np.int8), np.array([1, 1, 1, 1]))

    # Baseline: ball known + nearest_team -> H/A, unknown -> D.
    b = pd.DataFrame({"ball_x_m": [0.0, 0.0, np.nan, 1.0], "nearest_team": [0, 1, 0, -1]})
    assert list(hz.baseline_nearest(b)) == ["H", "A", "D", "D"]

    # run(): predictor never sees label or ball_dropped; predictor == GT scores 1.0.
    tabs = {m: table("DHHHAA", "AADDHH") for m in hz.ef.IDSSE_TUNE + hz.ef.IDSSE_HOLDOUT}
    loader = lambda mid, tables, seed: tabs[mid].copy()
    seen_cols = []

    def oracle(d):
        seen_cols.append(set(d.columns))
        return np.asarray(tabs[hz.ef.IDSSE_TUNE[0]]["label"]).astype(str)
    oracle.__name__ = "oracle"
    rows, summary = hz.run(oracle, "tune", "degraded", (0, 1), ids=hz.ef.IDSSE_TUNE[:1],
                           loader=loader)
    assert len(rows) == 2 and summary["spell_f1_mean"]["mean"] == 1.0, summary
    assert all(not ({"label", "ball_dropped"} & c) for c in seen_cols), seen_cols

    # Split guards.
    raises(ValueError, hz.split_ids, "tune", False, [hz.ef.IDSSE_HOLDOUT[0]])
    raises(ValueError, hz.run, oracle, "tune", "clean", (), ids=[hz.ef.IDSSE_HOLDOUT[1]],
           loader=loader)
    raises(PermissionError, hz.run, oracle, "holdout", "clean", (), loader=loader)
    raises(ValueError, hz.split_ids, "holdout", True, [hz.ef.IDSSE_TUNE[0]])
    raises(PermissionError, hz.load_table, hz.ef.IDSSE_HOLDOUT[0], "clean", None)

    # Tracked logs: every tuned key must cite a log frozen_config checks.
    cfg = hz.load_config()
    hz.check_tracked_logs(cfg)
    tuned = next(k for k, v in cfg["_provenance"].items() if v["basis"] == "tuned_external")
    for log in (None, {"path": "results/tuning/dead_ball_tune/crosscheck.json"}):
        bad = json.loads(json.dumps(cfg))
        bad["_provenance"][tuned].pop("tune_log")
        if log:
            bad["_provenance"][tuned]["tune_log"] = log
        raises(ValueError, hz.check_tracked_logs, bad)
    print("test_harness: all checks passed")


if __name__ == "__main__":
    main()
