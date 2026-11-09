"""What happens while the tactical ball track is missing? Label-free, aggregates only.

    python features/diagnose_ball_gaps.py          # after extract_features.py has been run

Reads data/derived/features_tactical.parquet and nothing else -- NO ground truth, no video -- and
prints only aggregate statistics (no per-frame or per-gap rows), so the output is safe to share.

Question it answers: are the long ball-missing stretches stoppages (players walking, a set piece
follows, ball last seen leaving the pitch) or live play the tracker lost (players running, ball
last seen in a crowded box)? Every statistic is shown side by side for
  GAP   frames inside a long ball-missing run, and
  SEEN  frames where the ball is tracked (the reference: mostly live play),
so no threshold has to be chosen to read it.

The constants below only bin and describe; nothing here feeds a model or a submission.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

FPS = 25
LONG_GAP_MIN_FRAMES = 126        # > 5 s: the bin holding 99.7% of missing frames on 2026-11-09
EDGE_FRAMES = 25                 # 1 s window just before a gap starts / after it ends
DURATION_BINS_S = [5, 10, 20, 40, 80, 160, 10 ** 6]
# Penalty area, IFAB Law 1: 16.5 m deep, 40.32 m wide, centred on the goal.
BOX_DEPTH_M, BOX_WIDTH_M = 16.5, 40.32


def long_gap_runs(df):
    """[(start_row, stop_row)] half-open, per graded half, for missing runs >= LONG_GAP_MIN_FRAMES.
    Runs touching the start or end of a half are kept but flagged, since they are cut off."""
    runs = []
    for half in (1, 2):
        rows = np.flatnonzero(df["half"].values == half)
        assert len(rows) and np.all(np.diff(rows) == 1), "half {} rows not contiguous".format(half)
        missing = (~df["ball_seen"].values[rows]).astype(np.int8)
        edges = np.diff(np.concatenate(([0], missing, [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a >= LONG_GAP_MIN_FRAMES:
                runs.append((rows[a], rows[0] + b, a == 0 or b == len(rows)))
    return runs


def in_box(x, y, length, width):
    on_pitch = (x >= 0) & (x <= length) & (y >= 0) & (y <= width)
    near_goal = on_pitch & ((x <= BOX_DEPTH_M) | (x >= length - BOX_DEPTH_M))
    central = np.abs(y - width / 2.0) <= BOX_WIDTH_M / 2.0
    return near_goal & central


def q(series):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return "   n/a"
    return "{:6.2f} {:6.2f} {:6.2f}".format(*s.quantile([0.25, 0.5, 0.75]).values)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", default=os.path.join(
        os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data"),
        "derived", "features_tactical.parquet"))
    args = parser.parse_args()

    length, width = GAMES[GAME]["pitch"]

    df = pd.read_parquet(args.features)
    graded = df["graded"].values
    runs = long_gap_runs(df)
    gap = np.zeros(len(df), dtype=bool)
    for a, b, _ in runs:
        gap[a:b] = True
    seen = graded & df["ball_seen"].values
    n_gap, n_seen = int(gap.sum()), int(seen.sum())
    assert n_gap > 0 and n_seen > 0

    print("long ball-missing runs (>= {:.1f} s): {}  covering {} frames ({:.1%} of graded)".format(
        LONG_GAP_MIN_FRAMES / FPS, len(runs), n_gap, n_gap / graded.sum()))
    print("  cut off by a half start/end: {}".format(sum(c for _, _, c in runs)))
    dur = np.array([(b - a) / FPS for a, b, _ in runs])
    counts, _ = np.histogram(dur, bins=DURATION_BINS_S)
    print("  duration (s):  " + "  ".join("{}-{}: {}".format(
        lo, hi if hi < 10 ** 6 else "inf", c) for lo, hi, c in
        zip(DURATION_BINS_S[:-1], DURATION_BINS_S[1:], counts)))
    print("  median {:.1f} s, longest {:.1f} s".format(np.median(dur), dur.max()))

    print("\n1) are players walking?  quartiles (25% / median / 75%)")
    print("  {:<34} {:>20}   {:>20}".format("", "GAP", "SEEN (reference)"))
    for col, label in (("all_walking_frac", "share of players walking"),
                       ("all_mean_speed", "mean player speed m/s"),
                       ("h_spread_m", "home spread m"), ("a_spread_m", "away spread m"),
                       ("n_tracked", "tracked players")):
        print("  {:<34} {:>20}   {:>20}".format(label, q(df[col].values[gap]),
                                                q(df[col].values[seen])))

    print("\n2) overlap with Smart Tagging (share of frames inside an active tag)")
    st = [c for c in df.columns if c.startswith("st_")]
    set_piece = [c for c in st if c.startswith("st_set_pieces")]
    any_sp = df[set_piece].any(axis=1).values if set_piece else np.zeros(len(df), bool)
    any_tag = df[st].any(axis=1).values
    print("  {:<34} {:>8.1%}   {:>8.1%}".format("any set-piece tag", any_sp[gap].mean(),
                                                any_sp[seen].mean()))
    print("  {:<34} {:>8.1%}   {:>8.1%}".format("any tag at all", any_tag[gap].mean(),
                                                any_tag[seen].mean()))
    print("  gaps with a set-piece tag starting inside or within 1 s after: {} / {}".format(
        sum(any_sp[a:min(b + EDGE_FRAMES, len(df))].any() for a, b, _ in runs), len(runs)))
    for c in sorted(st):
        share = df[c].values[gap].mean()
        if share >= 0.02:
            print("    {:<40} {:>6.1%} of gap frames   {:>6.1%} of seen".format(
                c, share, df[c].values[seen].mean()))

    print("\n3) the ball just before it vanished (last seen frame before each gap)")
    last = [a - 1 for a, b, cut in runs if not cut and df["ball_seen"].values[a - 1]]
    lx, ly = df["ball_x_m"].values[last], df["ball_y_m"].values[last]
    print("  gaps with a seen ball right before: {} / {}".format(len(last), len(runs)))
    if last:
        print("  last seen outside the pitch:       {:.0%}".format(
            np.mean(df["ball_out_m"].values[last] > 0)))
        print("  last seen in a penalty area:       {:.0%}   (share of SEEN frames in a box: "
              "{:.0%})".format(np.mean(in_box(lx, ly, length, width)),
                               np.mean(in_box(df["ball_x_m"].values[seen],
                                              df["ball_y_m"].values[seen], length, width))))
        # speed needs +/-k frames of track, so read it a few frames before the last seen frame
        print("  ball speed when last seen, m/s:    {}   (SEEN {})".format(
            q(df["ball_speed"].values[[i - 3 for i in last]]),
            q(df["ball_speed"].values[seen])))

    print("\n4) the ball when it came back (first frame after each gap)")
    first = [b for a, b, cut in runs if not cut and b < len(df) and df["ball_seen"].values[b]]
    print("  gaps with a seen ball right after: {} / {}".format(len(first), len(runs)))
    if first:
        fx, fy = df["ball_x_m"].values[first], df["ball_y_m"].values[first]
        print("  reappears outside the pitch:       {:.0%}".format(
            np.mean(df["ball_out_m"].values[first] > 0)))
        print("  reappears in a penalty area:       {:.0%}".format(
            np.mean(in_box(fx, fy, length, width))))
        print("  reappears within 9.15 m of the centre spot: {:.0%}".format(
            np.mean(np.hypot(fx - length / 2, fy - width / 2) <= 9.15)))
        print("  ball still over the next second:   {:.0%}".format(
            np.mean(df["ball_still_frac"].values[[min(i + 12, len(df) - 1) for i in first]]
                    >= 0.5)))
        pairs = [(a - 1, b) for a, b, cut in runs
                 if not cut and b < len(df) and df["ball_seen"].values[a - 1]
                 and df["ball_seen"].values[b]]
        if pairs:
            d = [np.hypot(df["ball_x_m"].values[j] - df["ball_x_m"].values[i],
                          df["ball_y_m"].values[j] - df["ball_y_m"].values[i]) for i, j in pairs]
            print("  distance between vanish and reappear points, m: {}".format(q(d)))


if __name__ == "__main__":
    main()
