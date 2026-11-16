"""Synthetic checks of features/dead_ball.py. Tiny made-up feature tables; no real files.

    .venv/bin/python features/tests/test_dead_ball.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import dead_ball as db  # noqa: E402

with open(os.path.join(HERE, "..", "config.json")) as f:
    CFG = json.load(f)
# Test values, independent of what config.json is later tuned to. Smoothing +/-2, lag 5.
CFG.update(dead_out_margin_m=0.61, dead_placed_still_frac_min=0.5, dead_vanish_offset_frames=0,
           dead_reappear_offset_frames=0, dead_decel_lag_frames=5, dead_onset_max_lookback_frames=125,
           dead_smooth_half_window_frames=2, dead_walking_frac_min=0.8,
           dead_mean_speed_max_m_s=1.5, dead_min_gap_frames=20, dead_min_alive_frames=10,
           dead_min_dead_frames=0)
CTX = {"known_dead": [], "smart_tagging": False}
RUN, STOP = (0.5, 3.0), (0.95, 0.8)  # (walking share, mean speed)


def table(n, halves=None):
    """Live play: ball known in the pitch, moving, home nearest, players running."""
    return pd.DataFrame({
        "half": np.ones(n, np.int8) if halves is None else np.asarray(halves, np.int8),
        "ball_x_m": np.full(n, 50.0), "ball_out_m": np.full(n, -5.0),
        "ball_still_frac": np.zeros(n), "nearest_team": np.zeros(n, np.int8),
        "h_walking_frac": np.full(n, RUN[0]), "a_walking_frac": np.full(n, RUN[0]),
        "all_walking_frac": np.full(n, RUN[0]), "all_mean_speed": np.full(n, RUN[1])})


def players(df, a, b, state):
    for c in ("h_walking_frac", "a_walking_frac", "all_walking_frac"):
        df.loc[a:b - 1, c] = state[0]
    df.loc[a:b - 1, "all_mean_speed"] = state[1]


def gap(df, a, b):
    df.loc[a:b - 1, ["ball_x_m", "ball_out_m", "ball_still_frac"]] = np.nan


def segs(df, cfg=CFG, ctx=CTX):
    out = db.detect(df, cfg, ctx)
    assert out["dead"].dtype == bool and len(out["dead"]) == len(df)
    for s, e in out["segments"]:
        assert out["dead"][s:e].all() and e - s > 1, (s, e)
    assert out["dead"].sum() == sum(e - s for s, e in out["segments"])
    assert len(out["restart_team"]) == len(out["segments"])
    return out


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError("{} did not raise {}".format(fn.__name__, exc.__name__))


def main():
    # 1. Scripted stoppage: ball seen crossing the line at 200, vanishes at 210, players slow from
    # 220 and walk, ball reappears moving at 400 -> D [200, 400): start = crossing, end = reappear.
    df = table(600)
    df.loc[200:209, "ball_out_m"] = 2.0
    gap(df, 210, 400)
    players(df, 220, 400, STOP)
    out = segs(df)
    assert out["segments"] == [(200, 400)], out["segments"]
    assert out["restart_team"] == ["H"], out["restart_team"]  # nearest player at the restart

    # 1b. No crossing: players stop at 190 (smoothed stopped run starts at 191), ball vanishes at
    # 210 -> start = 191 - lag 5 = 186.
    df = table(600)
    gap(df, 210, 400)
    players(df, 190, 400, STOP)
    assert segs(df)["segments"] == [(186, 400)]
    # 1b'. Players slow for 400 frames before the vanish at 500: run start 101 is more than 125
    # back -> clipped at 500 - 125 = 375, no lag. Run start exactly 125 back (375) -> lag applies.
    df = table(800)
    gap(df, 500, 700)
    players(df, 100, 700, STOP)
    assert segs(df)["segments"] == [(375, 700)], segs(df)["segments"]
    df = table(800)
    gap(df, 500, 700)
    players(df, 374, 700, STOP)               # smoothed run starts at 375
    assert segs(df)["segments"] == [(370, 700)], segs(df)["segments"]
    # 1e. Shipped tuned shape: no smoothing (h=0), lag 0 -> the edges are the raw frames.
    cfg0 = dict(CFG, dead_smooth_half_window_frames=0, dead_decel_lag_frames=0)
    df = table(600)
    gap(df, 210, 400)
    players(df, 190, 400, STOP)
    assert segs(df, cfg0)["segments"] == [(190, 400)], segs(df, cfg0)["segments"]
    df.loc[195, "all_mean_speed"] = RUN[1]      # one running frame: the run restarts at 196
    assert segs(df, cfg0)["segments"] == [(196, 400)], segs(df, cfg0)["segments"]
    df = table(600)
    gap(df, 210, 400)
    players(df, 300, 400, STOP)                 # stopping inside the gap: start = 300 exactly
    assert segs(df, cfg0)["segments"] == [(300, 400)], segs(df, cfg0)["segments"]
    # 1c. ... and with the players stopping only inside the gap, at 300 (run from 301): 296.
    df = table(600)
    gap(df, 210, 400)
    players(df, 300, 400, STOP)               # stopped on most of the gap, so its median passes
    assert segs(df)["segments"] == [(296, 400)], segs(df)["segments"]
    # 1d. Players stopped but no finite player data at all in the gap, no onset -> vanish frame.
    df = table(600)
    gap(df, 210, 400)
    df.loc[205:404, ["h_walking_frac", "a_walking_frac", "all_walking_frac", "all_mean_speed"]] = np.nan
    assert segs(df)["segments"] == [(210, 400)], segs(df)["segments"]

    # 2. Long live gap, players keep running -> no D at all.
    df = table(600)
    gap(df, 210, 400)
    out = segs(df)
    assert out["segments"] == [] and not out["dead"].any()
    # A short gap (< dead_min_gap_frames) with the players stopped is not a candidate.
    df = table(600)
    gap(df, 210, 225)
    players(df, 205, 230, STOP)
    assert segs(df)["segments"] == []

    # 3. Short alive blip (5 < 10 frames) between two ball-out candidates -> merged.
    df = table(600)
    df.loc[100:149, "ball_out_m"] = 2.0
    df.loc[155:199, "ball_out_m"] = 2.0
    assert segs(df)["segments"] == [(100, 200)]
    assert segs(df, dict(CFG, dead_min_alive_frames=3))["segments"] == [(100, 150), (155, 200)]
    # Ball within the margin of the line (0.5 m out < 0.61) is not out.
    df = table(600)
    df.loc[100:149, "ball_out_m"] = 0.5
    assert segs(df)["segments"] == []

    # 3b. dead_min_dead_frames: after merging, D segments shorter than it become alive. Two
    # out-of-play candidates 10 and 30 frames long; the blip-merged pair (100, 200) is 100 long.
    df = table(600)
    df.loc[300:309, "ball_out_m"] = 2.0
    df.loc[400:429, "ball_out_m"] = 2.0
    assert segs(df)["segments"] == [(300, 310), (400, 430)]           # 0 = off
    md = dict(CFG, dead_min_dead_frames=30)
    assert segs(df, md)["segments"] == [(400, 430)]                    # 10 < 30 dropped, 30 kept
    assert not segs(df, md)["dead"][300:310].any()
    assert segs(df, dict(CFG, dead_min_dead_frames=31))["segments"] == []
    df.loc[100:149, "ball_out_m"] = 2.0
    df.loc[155:199, "ball_out_m"] = 2.0                               # merged first, then length 100
    assert segs(df, dict(CFG, dead_min_dead_frames=60))["segments"] == [(100, 200)]
    kctx = {"known_dead": [(302, 305)], "smart_tagging": False}       # known_dead never dropped,
    assert segs(df, md, kctx)["segments"] == [(100, 200), (300, 310), (400, 430)]  # nor its segment
    raises(AssertionError, db.detect, df, dict(CFG, dead_min_dead_frames=-1), CTX)

    # 4. Adjacent periods (IDSSE): a stoppage across the seam at 300 splits there; the onset lag is
    # clipped at the block start; no merge across the seam.
    df = table(600, [1] * 300 + [2] * 300)
    gap(df, 250, 350)
    players(df, 200, 400, STOP)
    out = segs(df)
    assert out["segments"] == [(196, 300), (300, 350)], out["segments"]
    # alive tail of period 1 (5 frames) next to D at the start of period 2: not merged
    df = table(600, [1] * 300 + [2] * 300)
    df.loc[290:294, "ball_out_m"] = 2.0
    df.loc[300:340, "ball_out_m"] = 2.0
    assert segs(df)["segments"] == [(290, 295), (300, 341)], segs(df)["segments"]

    # 5. NaN team walking share -> all_walking_frac. Stopped via all_ -> D; running via all_ -> no D.
    df = table(600)
    gap(df, 210, 400)
    players(df, 190, 400, STOP)
    df.loc[150:450, "h_walking_frac"] = np.nan
    assert segs(df)["segments"] == [(186, 400)]
    df.loc[150:450, "all_walking_frac"] = RUN[0]
    assert segs(df)["segments"] == []

    # 6. Placed ball: ball reappears still at 400 and is kicked at 450, players stopped to 460 ->
    # the D segment ends when the ball leaves the still state.
    df = table(600)
    gap(df, 210, 400)
    players(df, 190, 460, STOP)
    df.loc[400:449, "ball_still_frac"] = 1.0
    assert segs(df)["segments"] == [(186, 450)], segs(df)["segments"]

    # 7. Smart Tagging: works without st_ columns (all above); with them, tags pick the team.
    df = table(600)
    gap(df, 210, 400)
    players(df, 190, 400, STOP)
    raises(AssertionError, db.detect, df, CFG, {"known_dead": [], "smart_tagging": True})
    st = {"known_dead": [], "smart_tagging": True}
    df["st_set_pieces_A"] = False
    df.loc[380:405, "st_set_pieces_A"] = True
    assert segs(df, ctx=st)["restart_team"] == ["A"]
    df["st_set_pieces_A"] = False
    df["st_set_pieces_NA"] = df["st_goals_H"] = False
    df.loc[380:405, "st_set_pieces_NA"] = True   # N/A falls through ...
    df.loc[250:300, "st_goals_H"] = True        # ... home scored -> away kicks off
    assert segs(df, ctx=st)["restart_team"] == ["A"]
    raises(AssertionError, db.detect, df, CFG, CTX)
    df = table(600)                              # nothing known at the end -> unknown
    gap(df, 210, 600)
    players(df, 190, 600, STOP)
    assert segs(df)["segments"] == [(186, 600)] and segs(df)["restart_team"] == ["unknown"]

    # 8. known_dead honoured to the frame, even in live play; must sit in one block.
    df = table(600, [0] * 100 + [2] * 500)
    out = segs(df, ctx={"known_dead": [(100, 133)], "smart_tagging": False})
    assert out["segments"] == [(100, 133)]
    raises(AssertionError, db.detect, df, CFG, {"known_dead": [(90, 133)], "smart_tagging": False})
    raises(AssertionError, db.detect, df, CFG, {"known_dead": []})
    raises(ValueError, db.detect, df.assign(label="H"), CFG, CTX)
    raises(ValueError, db.detect, df.assign(ball_dropped=False), CFG, CTX)
    for k in ("dead_decel_lag_frames", "dead_vanish_offset_frames", "dead_reappear_offset_frames",
              "dead_onset_max_lookback_frames"):
        raises(ValueError, db.detect, df, dict(CFG, **{k: -1}), CTX)
    raises(ValueError, db.detect, df.assign(st_goals_H=False), CFG,
           {"known_dead": [], "smart_tagging": True})   # st_ present but no st_set_pieces_*

    # 9. predict_labels: only H/A/D, right length; dead -> D; alive with unknown ball carries the
    # last alive team (short gap), else the next known team (block start), else D.
    df = table(600, [1] * 300 + [2] * 300)
    df.loc[0:49, "ball_x_m"] = np.nan          # block start unknown -> next known team (A)
    df.loc[50:99, "nearest_team"] = 1
    df.loc[100:299, "nearest_team"] = 0
    gap(df, 120, 130)                          # short live gap -> carried H
    df.loc[150:179, "ball_out_m"] = 2.0        # D
    lab = db.predict_labels(df, CFG, CTX)
    assert len(lab) == len(df) and set(lab) <= {"H", "A", "D"}
    assert (lab[:100] == "A").all() and (lab[100:150] == "H").all()
    assert (lab[150:180] == "D").all() and (lab[180:300] == "H").all()
    df.loc[300:599, "ball_x_m"] = np.nan       # period 2: nothing known, no carry across seam
    lab = db.predict_labels(df, CFG, CTX)
    assert (lab[300:] == "D").all() and (lab[180:300] == "H").all()

    # 10. Speed: one call on a 150k-row table with many gaps stays well under 2 s.
    n = 150_000
    rng = np.random.default_rng(0)
    df = table(n, [1] * (n // 2) + [2] * (n - n // 2))
    for a in rng.integers(0, n - 600, 150):
        gap(df, a, a + int(rng.integers(30, 500)))
    df["all_mean_speed"] = rng.uniform(0.5, 3.0, n)
    t0 = time.perf_counter()
    db.predict_labels(df, CFG, CTX)
    dt = time.perf_counter() - t0
    assert dt < 2.0, dt
    assert set(db.TUNABLE) <= set(CFG) and all(
        CFG["_provenance"][k]["basis"] == "tuned_external" and "revisit" not in CFG["_provenance"][k]
        for k in db.TUNABLE)
    print("test_dead_ball: all checks passed ({:.2f} s for {} rows)".format(dt, n))


if __name__ == "__main__":
    main()
