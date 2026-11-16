"""BAR-ATH submission: possession_decoder on the feature table -> data/derived/submission.csv.

    python features/extract_features.py        # first, once
    python features/make_submission.py

Reads data/derived/features_tactical.parquet (or $MTAH_DATA_DIR/derived/...) and
features/config.json -- no ground truth, no video. ctx is dead_ball_report.bar_ath_ctx (the 2nd-half
pre-kickoff span known dead, Smart Tagging present). Writes `frame,label`, one row per tactical
video frame, X on ungraded frames, checks it with the official validator and prints label counts
only. data/ is gitignored, so the file is never committed.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(HERE, "external"))
import harness  # noqa: E402
import possession_decoder  # noqa: E402
import possession_metrics as pm  # noqa: E402
from dead_ball_report import bar_ath_ctx  # noqa: E402
from games import GAME, GAMES  # noqa: E402


def main():
    derived = os.path.join(os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data"), "derived")
    cfg = harness.frozen_config()   # check_provenance + config equals the dead-ball tune logs
    df = pd.read_parquet(os.path.join(derived, "features_tactical.parquet"))
    n = GAMES[GAME]["n_video_frames"]
    assert len(df) == n, "feature table has {} rows, expected {}".format(len(df), n)
    assert (df["frame"].to_numpy() == np.arange(n)).all(), "feature rows not frames 0..{}".format(n - 1)
    lab = possession_decoder.decode(df, cfg, bar_ath_ctx(df, cfg))
    out = os.path.join(derived, "submission.csv")
    pd.DataFrame({"frame": np.arange(n), "label": lab}).to_csv(out, index=False)
    err = pm.validate_submission(pm.read_label_csv(out), n)
    if err:
        raise SystemExit("submission invalid: " + err)
    print("wrote {} ({} rows, valid)".format(out, n))
    graded = df["half"].to_numpy() > 0
    for k in ("H", "A", "D"):
        print("  {}: {} graded frames ({:.1%})".format(k, int((lab == k).sum()), (lab[graded] == k).mean()))
    print("  X: {} frames (ungraded)".format(int((lab == "X").sum())))


if __name__ == "__main__":
    main()
