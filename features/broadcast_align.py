"""Broadcast -> tactical frame alignment v1, from tracking only (docs/broadcast-alignment-design.md).

    python features/broadcast_align.py --report     # after extract_features.py has been run

DIAGNOSTIC ONLY: nothing in the decoder or the submission reads its output.

Reads the two ATDs (tactical + broadcast, positions only; no pixels, no homography, no labels),
data/derived/features_tactical.parquet for the dead-ball call, and features/config.json via
harness.frozen_config. Writes data/derived/broadcast_alignment.parquet, one row per broadcast frame
(row = broadcast ATD frame_id - 1):
    broadcast_frame, tactical_frame (-1 = none), segment_id (-1 = none), is_replay, match_dist_m
--report prints label-free aggregates only.

Method (core = align(), pure numpy on arrays):
  1. Player configurations per frame in metres, per team, slot identities ignored: the two ATDs
     are separate tracker runs (46 vs 51 slots) and slots are not persons (DATA.md).
     d(b, t) = median over the broadcast players of the distance to the nearest same-team tactical
     player at tactical frame t. One-directional, because the broadcast shows a partial view.
  2. Coarse: every coarse_step-th broadcast frame against every coarse_step-th tactical frame;
     the top_k local minima with d <= coarse_max_dist give candidate offsets (t - b).
  3. Refine: each broadcast frame with >= min_players tries the candidate offsets of its two
     bracketing coarse samples +/- refine_half_window frames; best t and d.
  4. Segments, each a constant offset: start at a frame whose best d <= max_dist, keep that offset
     while d(b, b + o) <= max_dist, the tactical frame stays in one half block (ef.block_spans) and
     players stay tracked; then refit the offset over the segment (+/- refine_half_window, capped
     distance sum) and split where the refitted d exceeds max_dist. Segments shorter than
     min_segment_frames are dropped (no blips); abutting segments with one offset are merged.
  5. Live chain: the max-total-length set of segments whose tactical intervals increase with the
     broadcast (t_start_j >= t_end_i). A segment outside it that starts before the live tactical
     time already reached is a replay (kept, flagged, never in the live map); any other is dropped.
Intervals are half-open [start, end).
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "external"))
import extract_features as ef  # noqa: E402
from games import GAME, GAMES, DATA_DIR, require, source_frames  # noqa: E402

KEYS = ("broadcast_align_coarse_step_frames", "broadcast_align_coarse_top_k",
        "broadcast_align_coarse_max_dist_m", "broadcast_align_refine_half_window_frames",
        "broadcast_align_min_players", "broadcast_align_max_dist_m",
        "broadcast_align_min_segment_frames")
CHUNK_PAIRS = 10000   # performance only: frame pairs per vectorised distance call
EXTEND_BLOCK = 250    # performance only: frames per vectorised segment-extension step
CHECK_EVERY = 10      # diagnostic only: orientation check uses every 10th coarse sample
LIVE, REPLAY, DROPPED = "live", "replay", "dropped"


# -- per-frame configurations --------------------------------------------------------------------

def compact(players, team, pitch):
    """players [N, S, >=2] normalised, team [S] in {0, 1} -> [N, 2, M, 2] metres float32, per team
    (0 home, 1 away), tracked players first, NaN padding. A slot counts if both x and y are finite."""
    assert players.ndim == 3 and players.shape[1] == len(team), (players.shape, len(team))
    assert set(np.unique(team)) <= {ef.HOME, ef.AWAY}
    x = players[:, :, 0].astype(np.float64) * pitch[0]
    y = players[:, :, 1].astype(np.float64) * pitch[1]
    bad = np.isnan(x) | np.isnan(y)
    parts = []
    for code in (ef.HOME, ef.AWAY):
        m = team == code
        order = np.argsort(bad[:, m], axis=1, kind="stable")
        xs = np.take_along_axis(np.where(bad, np.nan, x)[:, m], order, 1)
        ys = np.take_along_axis(np.where(bad, np.nan, y)[:, m], order, 1)
        parts.append(np.stack([xs, ys], -1))
    width = max(1, max(int((~np.isnan(p[..., 0])).sum(1).max(initial=0)) for p in parts))
    out = np.full((len(players), 2, width, 2), np.nan, dtype=np.float32)
    for k, p in enumerate(parts):
        w = min(width, p.shape[1])
        out[:, k, :w] = p[:, :w]
    return out


def n_players(pos):
    return (~np.isnan(pos[..., 0])).sum(axis=(1, 2))


FAR_M = 1e6  # pad coordinate for missing tactical players: never a nearest neighbour


def fill_far(tpos):
    """Tactical configurations with NaN padding replaced by FAR_M, for pair_dist."""
    return np.where(np.isnan(tpos), np.float32(FAR_M), tpos).astype(np.float32)


def pair_dist(B, T):
    """B [K, 2, Mb, 2] (NaN pad), T [K, 2, M, 2] (fill_far) -> [K]: median over B's players of the
    distance to the nearest same-team player of T (inf if T has no player of that team). Every B
    row needs >= 1 player."""
    assert not np.isnan(T).any(), "pair_dist needs fill_far(tpos)"
    dx = B[:, :, :, None, 0] - T[:, :, None, :, 0]
    dy = B[:, :, :, None, 1] - T[:, :, None, :, 1]
    dmin = np.sqrt((dx * dx + dy * dy).min(-1)).astype(np.float64)  # [K, 2, Mb], NaN at B pads
    dmin[dmin > FAR_M / 2] = np.inf
    dmin = dmin.reshape(len(B), -1)
    assert (~np.isnan(dmin)).any(axis=1).all(), "pair_dist on a broadcast frame without players"
    return np.nanmedian(dmin, axis=1)


def dists(bpos, tfar, bi, ti):
    """d(bi[j], ti[j]) for index arrays of equal length, chunked. tfar = fill_far(tpos)."""
    assert len(bi) == len(ti)
    out = np.empty(len(bi), dtype=np.float64)
    for a in range(0, len(bi), CHUNK_PAIRS):
        out[a:a + CHUNK_PAIRS] = pair_dist(bpos[bi[a:a + CHUNK_PAIRS]], tfar[ti[a:a + CHUNK_PAIRS]])
    return out


# -- stages --------------------------------------------------------------------------------------

def coarse_candidates(bpos, tfar, nb, nt, cfg):
    """[C, K] candidate offsets and validity for coarse broadcast samples 0, step, 2*step, ..."""
    step, k = cfg["broadcast_align_coarse_step_frames"], cfg["broadcast_align_coarse_top_k"]
    thr, minp = cfg["broadcast_align_coarse_max_dist_m"], cfg["broadcast_align_min_players"]
    bc = np.arange(0, len(bpos), step)
    tc = np.arange(0, len(tfar), step)
    tc = tc[nt[tc] >= minp]
    off = np.zeros((len(bc), k), dtype=np.int64)
    ok = np.zeros((len(bc), k), dtype=bool)
    for i, b in enumerate(bc):
        if nb[b] < minp or not len(tc):
            continue
        d = dists(bpos, tfar, np.full(len(tc), b), tc)
        left, right = np.r_[np.inf, d[:-1]], np.r_[d[1:], np.inf]
        cand = np.flatnonzero((d <= left) & (d <= right) & (d <= thr))  # local minima only
        cand = cand[np.argsort(d[cand], kind="stable")][:k]
        off[i, :len(cand)] = tc[cand] - b
        ok[i, :len(cand)] = True
    return off, ok


def refine(bpos, tfar, nb, off, ok, cfg):
    """Per broadcast frame: best tactical frame (-1) and its d (NaN) over the candidates."""
    step, w = cfg["broadcast_align_coarse_step_frames"], cfg["broadcast_align_refine_half_window_frames"]
    nt_frames = len(tfar)
    best_t = np.full(len(bpos), -1, dtype=np.int64)
    best_d = np.full(len(bpos), np.nan)
    frames = np.flatnonzero(nb >= cfg["broadcast_align_min_players"])
    r = np.arange(-w, w + 1)
    for a in range(0, len(frames), 2000):
        f = frames[a:a + 2000]
        c0 = f // step
        c1 = np.minimum(c0 + 1, len(off) - 1)
        o = np.concatenate([off[c0], off[c1]], 1)
        v = np.concatenate([ok[c0], ok[c1]], 1)
        t = (f[:, None, None] + o[:, :, None] + r).reshape(len(f), -1)
        v = np.repeat(v, len(r), axis=1) & (t >= 0) & (t < nt_frames)
        rows, cols = np.nonzero(v)
        D = np.full(t.shape, np.inf)
        D[rows, cols] = dists(bpos, tfar, f[rows], t[rows, cols])
        j = D.argmin(axis=1)
        has = np.isfinite(D[np.arange(len(f)), j])
        best_t[f[has]] = t[np.arange(len(f)), j][has]
        best_d[f[has]] = D[np.arange(len(f)), j][has]
    return best_t, best_d


def seg_dist(bpos, tfar, nb, tblock, frames, o, block, minp):
    """d(f, f + o) for frames; inf where t leaves the array or the half block, or f has < minp."""
    t = frames + o
    good = (t >= 0) & (t < len(tfar))
    good[good] &= tblock[t[good]] == block
    good &= nb[frames] >= minp
    d = np.full(len(frames), np.inf)
    d[good] = dists(bpos, tfar, frames[good], t[good])
    return d


def segments(bpos, tfar, nb, tblock, best_t, best_d, cfg):
    """[(b_start, b_end, offset)] half-open, constant-offset segments (step 4)."""
    thr, minp = cfg["broadcast_align_max_dist_m"], cfg["broadcast_align_min_players"]
    w, min_len = cfg["broadcast_align_refine_half_window_frames"], cfg["broadcast_align_min_segment_frames"]
    nb_frames = len(bpos)
    seed = best_d <= thr  # NaN -> False
    out = []
    b = 0
    while b < nb_frames:
        if not seed[b]:
            b += 1
            continue
        o = int(best_t[b] - b)
        block = tblock[best_t[b]]
        e = b + 1
        while e < nb_frames:  # extend while the offset keeps matching
            f = np.arange(e, min(e + EXTEND_BLOCK, nb_frames))
            fail = np.flatnonzero(seg_dist(bpos, tfar, nb, tblock, f, o, block, minp) > thr)
            if len(fail):
                e = int(f[fail[0]])
                break
            e = int(f[-1]) + 1
        frames = np.arange(b, e)
        # refit: offset minimising the capped distance sum over the segment
        cands = o + np.arange(-w, w + 1)
        costs = [np.minimum(seg_dist(bpos, tfar, nb, tblock, frames, c, block, minp), thr).sum()
                 for c in cands]
        o_fit = int(cands[int(np.argmin(costs))])
        good = seg_dist(bpos, tfar, nb, tblock, frames, o_fit, block, minp) <= thr
        edges = np.diff(np.concatenate(([0], good.astype(np.int8), [0])))
        for s, t in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if t - s >= min_len:
                out.append((b + int(s), b + int(t), o_fit))
        b = e
    # a seed offset 1-2 frames off can stop the extension early; the refit then restores the
    # offset, so abutting pieces with one offset are one segment
    merged = []
    for seg in out:
        if (merged and merged[-1][1] == seg[0] and merged[-1][2] == seg[2]
                and tblock[merged[-1][0] + seg[2]] == tblock[seg[0] + seg[2]]):
            merged[-1] = (merged[-1][0], seg[1], seg[2])
        else:
            merged.append(seg)
    for s, e, o in merged:   # step 4: a segment never spans a half block
        assert tblock[s + o] == tblock[e - 1 + o], "segment {}-{} spans a half block".format(s, e)
    return merged


def classify(segs):
    """Kind per segment (step 5). segs sorted by b_start and disjoint in b."""
    n = len(segs)
    if not n:
        return []
    length = np.array([e - s for s, e, _ in segs], dtype=np.int64)
    t0 = np.array([s + o for s, _, o in segs]); t1 = np.array([e + o for _, e, o in segs])
    score, prev = length.copy(), np.full(n, -1)
    # ponytail: O(n^2) chain DP over segments; fine for thousands, sort-based LIS if ever needed
    for j in range(n):
        for i in range(j):
            if t1[i] <= t0[j] and score[i] + length[j] > score[j]:
                score[j], prev[j] = score[i] + length[j], i
    chain = set()
    j = int(np.argmax(score))
    while j >= 0:
        chain.add(j)
        j = int(prev[j])
    kinds, reached = [], -np.inf
    for j in range(n):
        if j in chain:
            kinds.append(LIVE)
            reached = max(reached, t1[j])
        else:
            kinds.append(REPLAY if t0[j] < reached else DROPPED)
    return kinds


def align(bpos, tpos, tblock, cfg):
    """Core, pure on arrays. bpos [Nb, 2, Mb, 2], tpos [Nt, 2, M, 2] metres (compact), tblock [Nt]
    half-block id per tactical frame. Returns dict of per-broadcast-frame arrays + segment table."""
    for k in KEYS:
        assert k in cfg, "config lacks {}".format(k)
    assert cfg["broadcast_align_min_players"] >= 1 and cfg["broadcast_align_coarse_step_frames"] >= 1
    assert cfg["broadcast_align_max_dist_m"] <= cfg["broadcast_align_coarse_max_dist_m"]
    assert bpos.ndim == 4 and bpos.shape[1:2] == (2,) and bpos.shape[3] == 2, bpos.shape
    assert tpos.ndim == 4 and tpos.shape[1:2] == (2,) and tpos.shape[3] == 2, tpos.shape
    assert tblock.shape == (len(tpos),)
    nb, nt = n_players(bpos), n_players(tpos)
    tfar = fill_far(tpos)  # every stage below takes the filled array
    off, ok = coarse_candidates(bpos, tfar, nb, nt, cfg)
    best_t, best_d = refine(bpos, tfar, nb, off, ok, cfg)
    segs = segments(bpos, tfar, nb, tblock, best_t, best_d, cfg)
    kinds = classify(segs)

    n = len(bpos)
    tac = np.full(n, -1, dtype=np.int64)
    seg_id = np.full(n, -1, dtype=np.int32)
    replay = np.zeros(n, dtype=bool)
    dist = best_d.copy()
    table = []
    for (s, e, o), kind in zip(segs, kinds):
        if kind == DROPPED:
            continue
        sid = len(table)
        table.append({"segment_id": sid, "b_start": s, "b_end": e, "offset": o, "kind": kind})
        f = np.arange(s, e)
        tac[s:e] = f + o
        seg_id[s:e] = sid
        replay[s:e] = kind == REPLAY
        dist[s:e] = dists(bpos, tfar, f, f + o)
    live = (tac >= 0) & ~replay
    assert (np.diff(tac[live]) > 0).all(), "live mapping not strictly increasing"
    assert ((tac[tac >= 0] >= 0) & (tac[tac >= 0] < len(tpos))).all()
    assert all(r["b_end"] - r["b_start"] >= cfg["broadcast_align_min_segment_frames"] for r in table)
    return {"tactical_frame": tac, "segment_id": seg_id, "is_replay": replay,
            "match_dist_m": dist, "n_players": nb, "segments": table,
            "n_dropped_segments": kinds.count(DROPPED)}


def orientation_check(bpos, tpos, pitch, cfg):
    """Label-free: share of sampled coarse broadcast samples with a coarse match, for the
    broadcast as given, teams swapped, y mirrored and pitch rotated 180 deg. Selects nothing."""
    step, minp = cfg["broadcast_align_coarse_step_frames"], cfg["broadcast_align_min_players"]
    nb, nt = n_players(bpos), n_players(tpos)
    tfar = fill_far(tpos)
    bc = np.arange(0, len(bpos), step * CHECK_EVERY)
    bc = bc[nb[bc] >= minp]
    tc = np.arange(0, len(tpos), step)
    tc = tc[nt[tc] >= minp]
    L, W = pitch
    variants = {"as given": bpos, "teams swapped": bpos[:, ::-1],
                "y mirrored": bpos * np.array([1, -1], np.float32) + np.array([0, W], np.float32),
                "rotated 180": bpos * np.array([-1, -1], np.float32) + np.array([L, W], np.float32)}
    out = {}
    for name, p in variants.items():
        hits = [dists(p, tfar, np.full(len(tc), b), tc).min() <= cfg["broadcast_align_coarse_max_dist_m"]
                for b in bc] if len(tc) else []
        out[name] = float(np.mean(hits)) if hits else float("nan")
    return out, len(bc)


# -- aggregates (label-free) ---------------------------------------------------------------------

def diagnostics(res, graded, dead, tac_ball, b_ball):
    """tac_ball [Nt, 2], b_ball [Nb, 2] metres, NaN = not seen; graded, dead [Nt] bool (dead may be
    None). Aggregates only."""
    tac, replay = res["tactical_frame"], res["is_replay"]
    live = (tac >= 0) & ~replay
    inv = np.full(len(graded), -1, dtype=np.int64)
    inv[tac[live]] = np.flatnonzero(live)
    t_seen = ~np.isnan(tac_ball).any(1)
    b_seen = ~np.isnan(b_ball).any(1)
    covered = inv >= 0
    b_has = np.zeros(len(graded), dtype=bool)
    b_has[covered] = b_seen[inv[covered]]
    miss = graded & ~t_seen
    out = {"tac_ball_missing_graded": int(miss.sum()),
           "of_which_live_mapped": int((miss & covered).sum()),
           "of_which_broadcast_ball_seen": int((miss & b_has).sum())}
    if dead is not None:
        for name, m in (("dead", dead), ("alive", ~dead)):
            out["missing_{}".format(name)] = int((miss & m).sum())
            out["missing_{}_broadcast_ball_seen".format(name)] = int((miss & m & b_has).sum())
    both = covered & t_seen & b_has
    d = np.hypot(*(tac_ball[both] - b_ball[inv[both]]).T)
    out["ball_both_seen_frames"] = int(both.sum())
    out["ball_dist_median_m"] = float(np.median(d)) if len(d) else float("nan")
    out["ball_dist_p90_m"] = float(np.percentile(d, 90)) if len(d) else float("nan")
    return out


def summary(res, tblock, cfg):
    tac, replay, segs = res["tactical_frame"], res["is_replay"], res["segments"]
    n = len(tac)
    live = (tac >= 0) & ~replay
    lives = [s for s in segs if s["kind"] == LIVE]
    offs = np.repeat([s["offset"] for s in lives], [s["b_end"] - s["b_start"] for s in lives])
    d = res["match_dist_m"][tac >= 0]
    out = {"broadcast_frames": n,
           "share_any_player": float(np.mean(res["n_players"] >= 1)),
           "share_min_players": float(np.mean(res["n_players"] >= cfg["broadcast_align_min_players"])),
           "share_live": float(live.mean()), "share_replay": float(replay.mean()),
           "share_unmapped": float((tac < 0).mean()),
           "live_segments": len(lives), "replay_segments": sum(s["kind"] == REPLAY for s in segs),
           "dropped_segments": res["n_dropped_segments"],
           "live_monotone": bool((np.diff(tac[live]) > 0).all()),
           "match_dist_median_m": float(np.median(d)) if len(d) else float("nan"),
           "match_dist_p90_m": float(np.percentile(d, 90)) if len(d) else float("nan")}
    if len(offs):
        out.update(offset_min=int(offs.min()), offset_median=float(np.median(offs)),
                   offset_max=int(offs.max()), offset_distinct=int(len(np.unique(offs))))
        by_block = {}
        for s in lives:
            by_block.setdefault(int(tblock[s["b_start"] + s["offset"]]), set()).add(s["offset"])
        out["distinct_live_offsets_per_tactical_block"] = {k: len(v) for k, v in sorted(by_block.items())}
    return out


# -- driver --------------------------------------------------------------------------------------

def frame_rate(meta_path):
    node = ET.parse(meta_path).getroot().find(".//GlobalConfig/FrameRate")
    assert node is not None, "{} has no GlobalConfig/FrameRate".format(meta_path)
    return float(node.text)


def main():
    import harness  # noqa: E402  features/external
    import dead_ball as db
    from dead_ball_report import bar_ath_ctx

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="print label-free aggregates")
    args = parser.parse_args()
    cfg = harness.frozen_config()
    pitch = GAMES[GAME]["pitch"]
    derived = os.path.join(DATA_DIR, "derived")
    arrays = {}
    for source in ("tactical", "broadcast"):
        meta, raw = require(["atd_meta", "atd_raw"], GAME, None, source)
        fps = frame_rate(meta)
        assert fps == cfg["fps"], "{} ATD FrameRate {} != config fps {}".format(source, fps, cfg["fps"])
        team, names, players, ball = ef.load_arrays(
            meta, raw, source_frames(source, GAME),
            os.path.join(derived, "atd_{}_cache.npz".format(source)), True)
        arrays[source] = (team, names, players, ball)
    tt, tnames, tp, tb = arrays["tactical"]
    bt, bnames, bp, bb = arrays["broadcast"]
    print("fps {} both; team names agree (home, away): {}".format(cfg["fps"], tnames == bnames))

    half = ef.half_index(GAME, len(tp))
    tblock = ef.block_ids(ef.block_spans(half), len(half))
    tpos, bpos = compact(tp, tt, pitch), compact(bp, bt, pitch)
    del tp, bp
    res = align(bpos, tpos, tblock, cfg)

    out = os.path.join(derived, "broadcast_alignment.parquet")
    pd.DataFrame({"broadcast_frame": np.arange(len(bpos), dtype=np.int32),
                  "tactical_frame": res["tactical_frame"].astype(np.int32),
                  "segment_id": res["segment_id"], "is_replay": res["is_replay"],
                  "match_dist_m": res["match_dist_m"].astype(np.float32)}).to_parquet(out, index=False)
    print("wrote {}".format(out))
    if not args.report:
        return

    df = pd.read_parquet(os.path.join(derived, "features_tactical.parquet"))
    assert len(df) == len(tpos) and (df["half"].to_numpy() == half).all(), "feature table mismatch"
    dead = db.detect(df, cfg, bar_ath_ctx(df, cfg))["dead"]
    scale = np.array(pitch)
    diag = diagnostics(res, half > 0, dead, tb[:, :2].astype(np.float64) * scale,
                       bb[:, :2].astype(np.float64) * scale)
    check, n_check = orientation_check(bpos, tpos, pitch, cfg)
    print("\nalignment summary (aggregates only)")
    for k, v in summary(res, tblock, cfg).items():
        print("  {:<42} {}".format(k, "{:.1%}".format(v) if k.startswith("share") else v))
    print("tactical graded frames with the tactical ball missing")
    for k, v in diag.items():
        print("  {:<42} {}".format(k, "{:.3f}".format(v) if isinstance(v, float) else v))
    print("orientation check: coarse match share over {} sampled broadcast samples".format(n_check))
    for k, v in check.items():
        print("  {:<42} {:.1%}".format(k, v))


if __name__ == "__main__":
    main()
