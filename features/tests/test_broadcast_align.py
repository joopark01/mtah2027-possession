"""Synthetic checks of features/broadcast_align.py. Made-up arrays with known answers; no files.

    .venv/bin/python features/tests/test_broadcast_align.py

Tactical: 6000 frames, two half blocks [0, 3000) and [3000, 6000), 11 + 11 players on smooth
paths (46 slots, some empty, one player dropping out now and then), everyone still on
[4000, 4300) (a stoppage: per-frame matching is ambiguous there, the segment must hold its offset). Broadcast: 5400 frames, a
partial view (players within 25 m of the play's centre in x), 0.3 m noise, its own slot order:
    [0, 800)      live, t = b + 200
    [800, 900)    close-up: no players
    [900, 1600)   live, same offset (a cut keeps it)
    [1600, 1800)  replay of t [1000, 1200)
    [1800, 2800)  live, t = b + 200
    [2800, 2860)  3 players only (below min_players)
    [2860, 5400)  live in the second half, t = b + 400
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import broadcast_align as ba  # noqa: E402
import extract_features as ef  # noqa: E402

with open(os.path.join(HERE, "..", "config.json")) as f:
    CFG = json.load(f)
ef.check_provenance(CFG)
# Test values, independent of what config.json holds.
CFG.update(broadcast_align_coarse_step_frames=25, broadcast_align_coarse_top_k=3,
           broadcast_align_coarse_max_dist_m=3.0, broadcast_align_refine_half_window_frames=12,
           broadcast_align_min_players=6, broadcast_align_max_dist_m=1.5,
           broadcast_align_min_segment_frames=25)
L, W = 105.3, 68.0
NT, NB = 6000, 5400
PIECES = [(0, 800, "live", 200), (800, 900, "closeup", None), (900, 1600, "live", 200),
          (1600, 1800, "replay", -600), (1800, 2800, "live", 200), (2800, 2860, "few", 200),
          (2860, 5400, "live", 400)]


def tactical(rng):
    """players [NT, 46, 3] normalised, team [46], ball [NT, 2] metres."""
    t = np.arange(NT)[:, None]
    team = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    slots = np.r_[0:11, 23:34]
    pos = np.full((NT, 46, 3), np.nan)
    for axis, size in ((0, L), (1, W)):
        c = rng.uniform(0.2, 0.8, 22) * size
        val = c.copy()
        for _ in range(2):  # two incommensurate sinusoids per axis, ~2-4 m/s
            amp, per = rng.uniform(4, 10, 22), rng.uniform(300, 900, 22)
            val = val + amp * np.sin(2 * np.pi * t / per + rng.uniform(0, 2 * np.pi, 22))
        pos[:, slots, axis] = np.clip(val, 0, size) / size
    pos[:, slots, 2] = 3.0
    pos[::37, slots[5], :] = np.nan  # a player dropping out now and then
    pos[4000:4300] = pos[4000]          # a stoppage: everyone still for 12 s (broadcast b 3600-3900)
    ball = np.stack([L / 2 + 20 * np.sin(np.arange(NT) / 150), W / 2 + 10 * np.cos(np.arange(NT) / 90)], 1)
    return pos, team, ball


def broadcast(pos, team, ball, rng):
    """players [NB, 51, 3] normalised, team [51], ball [NB, 2] metres, truth t [NB] and kind [NB]."""
    bteam = np.array([0] * 25 + [1] * 26, dtype=np.int8)
    out = np.full((NB, 51, 3), np.nan)
    bball = np.full((NB, 2), np.nan)
    truth = np.full(NB, -1)
    kind = np.empty(NB, dtype=object)
    for s, e, k, o in PIECES:
        kind[s:e] = k
        if k == "closeup":
            continue
        b = np.arange(s, e)
        t = b + o
        truth[s:e] = t if k in ("live", "replay") else -1
        x = pos[t, :, 0] * L
        centre = np.nanmean(x, axis=1, keepdims=True)
        vis = np.abs(x - centre) < 25.0
        if k == "few":
            vis &= np.cumsum(vis, axis=1) <= 3  # 3 visible players in all
        for code in (0, 1):  # place each team's visible players into shuffled broadcast slots
            src = np.flatnonzero(team == code)
            dst = rng.permutation(np.flatnonzero(bteam == code))[:len(src)]
            for a, c in zip(src, dst):
                m = vis[:, a] & ~np.isnan(pos[t, a, 0])
                noise = rng.normal(0, 0.3, (len(b), 2)) / np.array([L, W])
                out[b[m], c, :2] = pos[t[m], a, :2] + noise[m]
                out[b[m], c, 2] = 3.0
    return out, bteam, bball, truth, kind


def main():
    rng = np.random.default_rng(7)
    pos, team, tball = tactical(rng)
    bp, bteam, bball, truth, kind = broadcast(pos, team, tball, rng)
    half = np.r_[np.ones(3000, np.int8), np.full(3000, 2, np.int8)]
    tblock = ef.block_ids(ef.block_spans(half), NT)

    # compact: shapes, padding, metres
    tpos, bpos = ba.compact(pos, team, (L, W)), ba.compact(bp, bteam, (L, W))
    assert tpos.shape == (NT, 2, 11, 2), tpos.shape
    assert ba.n_players(tpos)[0] == 21 and ba.n_players(tpos)[1] == 22  # frame 0: slot dropped
    assert np.isclose(tpos[1, 0, 0, 0], pos[1, 0, 0] * L)
    assert (ba.n_players(bpos)[800:900] == 0).all() and (ba.n_players(bpos)[2800:2860] <= 6).all()
    live_truth = kind == "live"
    assert (ba.n_players(bpos)[live_truth | (kind == "replay")] >= 6).all(), "fixture: view too narrow"

    res = ba.align(bpos, tpos, tblock, CFG)
    tac, rep, sid = res["tactical_frame"], res["is_replay"], res["segment_id"]

    # live frames: every one mapped, to the frame, and not flagged replay
    bad = np.flatnonzero(live_truth & ((tac != truth) | rep))
    assert not len(bad), "live frames wrong: {} (first {})".format(len(bad), bad[:5])
    # replay: flagged, mapped to the replayed frames, never in the live map
    r = kind == "replay"
    assert rep[r].all() and (tac[r] == truth[r]).all(), "replay not recovered"
    assert not rep[~r].any(), "spurious replay frames"
    # close-up and too-few-players stretches unmapped, to the frame
    for k in ("closeup", "few"):
        m = kind == k
        assert (tac[m] == -1).all() and (sid[m] == -1).all(), "{} mapped".format(k)
    # segments: 4 live, 1 replay, boundaries exact, none shorter than the minimum
    segs = res["segments"]
    got = [(s["b_start"], s["b_end"], s["kind"], s["offset"]) for s in segs]
    want = [(a, b, k, o) for a, b, k, o in PIECES if k in ("live", "replay")]
    assert got == want, got
    live = (tac >= 0) & ~rep
    assert (np.diff(tac[live]) > 0).all()
    s = ba.summary(res, tblock, CFG)
    assert s["live_segments"] == 4 and s["replay_segments"] == 1 and s["live_monotone"]
    assert s["distinct_live_offsets_per_tactical_block"] == {0: 1, 1: 1}
    assert abs(s["share_unmapped"] - 160 / NB) < 1e-9 and s["match_dist_median_m"] < 0.6
    print("alignment recovered: {} live segments, 1 replay, {} unmapped frames".format(
        s["live_segments"], int((tac < 0).sum())))

    # diagnostics: tactical ball missing on t [1000, 1100) (live = close-up, replay shows it: not
    # counted) and [2200, 2400) (live b [2000, 2200), broadcast ball seen); dead on [2200, 2300)
    tb = tball.copy()
    tb[1000:1100] = np.nan
    tb[2200:2400] = np.nan
    bb = np.full((NB, 2), np.nan)
    bb[1600:2800] = tball[truth[1600:2800]] + np.array([0.4, 0.3])  # 0.5 m off where both seen
    dead = np.zeros(NT, dtype=bool)
    dead[2200:2300] = True
    graded = np.ones(NT, dtype=bool)
    d = ba.diagnostics(res, graded, dead, tb, bb)
    assert d["tac_ball_missing_graded"] == 300 and d["of_which_live_mapped"] == 200
    assert d["of_which_broadcast_ball_seen"] == 200
    assert d["missing_dead"] == 100 and d["missing_dead_broadcast_ball_seen"] == 100
    assert d["missing_alive"] == 200 and d["missing_alive_broadcast_ball_seen"] == 100
    assert d["ball_both_seen_frames"] == 800, d  # b [1800, 2800) minus the 200 missing
    assert abs(d["ball_dist_median_m"] - 0.5) < 1e-6 and abs(d["ball_dist_p90_m"] - 0.5) < 1e-6
    print("diagnostics ok: {}".format(d))

    # no blips: per-frame jitter must not leave short segments
    assert all(s["b_end"] - s["b_start"] >= 25 for s in segs)

    # orientation check: a rotated broadcast matches only as "rotated 180"
    rot = bpos * np.array([-1, -1], np.float32) + np.array([L, W], np.float32)
    share, n = ba.orientation_check(rot, tpos, (L, W), CFG)
    assert n > 0 and share["rotated 180"] > 0.9 and share["as given"] < 0.2, share
    share, _ = ba.orientation_check(bpos, tpos, (L, W), CFG)
    assert share["as given"] > 0.9 and share["teams swapped"] < 0.2, share
    print("orientation check ok: {}".format(share))

    # a shot of 20 frames (< min segment) between close-ups is dropped, not a blip
    bpos2 = bpos.copy()
    bpos2[3000:3100] = np.nan
    bpos2[3040:3060] = bpos[3040:3060]
    res2 = ba.align(bpos2, tpos, tblock, CFG)
    assert (res2["tactical_frame"][3000:3100] == -1).all()
    assert (res2["tactical_frame"][3100:3200] == truth[3100:3200]).all()
    print("short shot dropped ok")
    print("all broadcast_align tests passed")


if __name__ == "__main__":
    main()
