"""Dead-ball detector on BAR-ATH: label-free plausibility aggregates only.

    python features/dead_ball_report.py          # after extract_features.py has been run

Reads data/derived/features_tactical.parquet and features/config.json -- NO ground truth, no
video -- runs dead_ball.detect and prints only aggregates (no per-frame or per-segment rows), so
the output is safe to share. Design doc section 6: D share, stoppages per half, shortest D
segment, restart-team mix vs Smart Tagging. Nothing here feeds back into any parameter.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import dead_ball as db  # noqa: E402
import extract_features as ef  # noqa: E402

SHORT_FRAMES = 25  # 1 s: design doc 6 expects no D segment shorter than ~1 s


def main():
    path = os.path.join(os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data"),
                        "derived", "features_tactical.parquet")
    with open(os.path.join(HERE, "config.json")) as f:
        cfg = json.load(f)
    ef.check_provenance(cfg)
    df = pd.read_parquet(path)
    half = df["half"].to_numpy()
    # published: the 2nd half is graded from its first row, the kickoff is later (DATA.md, FAQ.md)
    first2, ko2 = int(np.flatnonzero(half == 2)[0]), int(cfg["kickoff_video_frames"]["2"])
    if not first2 < ko2:
        raise ValueError("2nd half starts at row {}, not before its kickoff {}".format(first2, ko2))
    ctx = {"known_dead": [(first2, ko2)], "smart_tagging": True}
    out = db.detect(df, cfg, ctx)
    graded = df["graded"].to_numpy()
    dead = out["dead"]
    print("graded frames {}, predicted D share {:.1%}".format(graded.sum(), dead[graded].mean()))
    # IDSSE periods start at the kickoff, so compare with IDSSE without the pre-kickoff span
    ko = graded.copy()
    ko[first2:ko2] = False
    print("  excluding the 2nd-half pre-kickoff span: D share {:.1%}, {} D segments".format(
        dead[ko].mean(), sum(ko[s] for s, _ in out["segments"])))
    segs = [(s, e, t) for (s, e), t in zip(out["segments"], out["restart_team"]) if half[s] in (1, 2)]
    lens = np.array([e - s for s, e, _ in segs])
    for h in (1, 2):
        n = sum(half[s] == h for s, _, _ in segs)
        print("  half {}: {} D segments".format(h, n))
    print("  total {} D segments (design doc: a match has roughly 100-120 restarts)".format(len(segs)))
    print("  length s: shortest {:.2f}, quartiles {:.1f} / {:.1f} / {:.1f}, longest {:.1f}".format(
        lens.min() / 25, *(np.percentile(lens, [25, 50, 75]) / 25), lens.max() / 25))
    print("  shorter than {:.0f} s: {} ({:.1%})".format(SHORT_FRAMES / 25, (lens < SHORT_FRAMES).sum(),
                                                       np.mean(lens < SHORT_FRAMES)))
    teams = pd.Series([t for _, _, t in segs]).value_counts()
    print("  restart team: " + ", ".join("{} {}".format(k, v) for k, v in teams.items()))
    sp = df[[c for c in df.columns if c.startswith("st_set_pieces")]].any(axis=1).to_numpy()
    print("  D segments overlapping any set-piece tag, any team incl. N/A (or within 1 s after): {} / {}".format(
        sum(sp[s:min(e + 25, len(df))].any() for s, e, _ in segs), len(segs)))
    seen = df["ball_seen"].to_numpy()
    print("  D frames with the ball seen: {:.1%};  ball-missing graded frames called D: {:.1%}".format(
        seen[dead & graded].mean(), dead[graded & ~seen].mean()))


if __name__ == "__main__":
    main()
