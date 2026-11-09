"""Per-frame feature table from the raw tactical ATD (+ Smart Tagging), for possession modelling.

    pip install -r features/requirements.txt
    python features/extract_features.py                    # writes data/derived/features_tactical.parquet
    python features/extract_features.py --report           # + label-free sanity summary

Runs LOCALLY on the gated data. The output is frame-level derived data and lives under data/,
which is gitignored -- do not commit it or upload it anywhere (RULES.md, terms of use).

What it produces: one row per video frame (199,319), indexed by `frame` = submission row.

  meta     frame, half (0 = not in a graded half), graded
  ball     raw + gap-interpolated position (m), visibility/gap bookkeeping, velocity, acceleration,
           turn angle, stillness, distance outside the pitch, any extra raw ball fields as-is
  teams    for h_ (home, Barcelona) and a_ (away, Athletic): track count, centroid, spread,
           speeds, walking share, attack-direction-normalised position and velocity, share in own
           half, nearest/second-nearest distance to the ball, players near the ball
  cross    nearest team to the ball and the margin, free space around the nearest player,
           all-player speed / walking share
  st_      Smart Tagging: one boolean per (code, team) that is active on the frame

Conventions, stated once:
  * Positions in metres: normalised ATD x is along the pitch LENGTH, y along the WIDTH, scaled by
    games.GAMES[game]["pitch"] = (105.3, 68.0). Orientation is checked, not assumed: at each
    kickoff the two team centroids must be separated along x.
  * Velocities by central difference over +/- velocity_half_window_frames on the (interpolated)
    track, in m/s. NaN wherever either end is missing.
  * "attack sign": +1 if that team attacks toward +x in that half. Read off the kickoff, where
    every player stands in their own half by law.
  * Nothing is interpolated across a half boundary.
  * Every threshold comes from features/config.json. There are no defaults in this file.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, half_video_blocks, require, source_frames  # noqa: E402

FIELD_SEP, GROUP_SEP, VALUE_SEP = ":", ";", ","
HOME, AWAY = 0, 1


# -- raw parsing --------------------------------------------------------------------------------

def read_meta(meta_path):
    """(team of each slot [S] in {0,1}, (home_name, away_name)). First Team in the file is home
    (FIFA/EPTS convention, same rule as starter/baseline.py)."""
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    assert len(teams) == 2, "expected 2 teams in {}, found {}".format(meta_path, len(teams))
    home_id = teams[0][0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    assert slots, "no Player entries in {}".format(meta_path)
    unknown = set(slots) - {teams[0][0], teams[1][0]}
    assert not unknown, "Player teamId not among the two teams: {}".format(unknown)
    team = np.array([HOME if s == home_id else AWAY for s in slots], dtype=np.int8)
    return team, (teams[0][1], teams[1][1])


def _parse_group(group, n_fields, out_row):
    """Fill out_row[:n_fields] from 'x,y,s[,...]'. Missing observation -> stays NaN."""
    if group[0] == "N":  # "NaN" or "NaN,NaN,NaN" -- missing
        return
    values = group.split(VALUE_SEP)
    assert len(values) == n_fields, "group {!r} has {} fields, expected {}".format(
        group, len(values), n_fields)
    for k in range(n_fields):
        out_row[k] = float(values[k])


def parse_raw_atd(raw_path, n_slots, n_frames):
    """Stream the raw ATD into arrays, failing on anything unexpected.

    Returns players [N, S, 3] (x_norm, y_norm, speed_m_s) and ball [N, B] (raw fields, B
    discovered from the first visible ball), both float32, NaN where missing. Row i = video frame i
    (ATD frame_id - 1), and the ids must be exactly 1..N.
    """
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = None
    n_ball_fields = None
    expected_id = 1
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            assert len(parts) == 3, "line for frame_id {} has {} ':'-parts, expected 3".format(
                expected_id, len(parts))
            frame_id = int(parts[0])
            assert frame_id == expected_id, "frame_id {} where {} expected -- ATD not contiguous " \
                "from 1".format(frame_id, expected_id)
            row = frame_id - 1
            groups = parts[1].split(GROUP_SEP)
            assert len(groups) == n_slots, "frame_id {}: {} player groups, metadata has {} " \
                "slots".format(frame_id, len(groups), n_slots)
            for slot, group in enumerate(groups):
                _parse_group(group, 3, players[row, slot])

            ball_group = parts[2]
            if ball_group[0] != "N":
                if n_ball_fields is None:
                    n_ball_fields = len(ball_group.split(VALUE_SEP))
                    assert n_ball_fields >= 2, "ball group {!r} has <2 fields".format(ball_group)
                    ball = np.full((n_frames, n_ball_fields), np.nan, dtype=np.float32)
                _parse_group(ball_group, n_ball_fields, ball[row])
            expected_id += 1
    assert expected_id - 1 == n_frames, "{} ATD records, video has {}".format(
        expected_id - 1, n_frames)
    assert ball is not None, "the ball is never visible -- wrong file?"
    return players, ball


def file_fingerprint(path):
    st = os.stat(path)
    return {"path": os.path.abspath(path), "bytes": st.st_size, "mtime": int(st.st_mtime)}


def load_arrays(meta_path, raw_path, n_frames, cache_path, use_cache):
    """Parse once, then reuse the .npz cache as long as the raw file is unchanged."""
    team, names = read_meta(meta_path)
    fp = file_fingerprint(raw_path)
    if use_cache and os.path.isfile(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if (int(cached["raw_bytes"]) == fp["bytes"] and int(cached["raw_mtime"]) == fp["mtime"]
                and cached["players"].shape == (n_frames, len(team), 3)):
            print("  using cache {}".format(os.path.relpath(cache_path, ROOT)))
            return team, names, cached["players"], cached["ball"]
        print("  cache is stale, re-parsing")
    t0 = time.time()
    players, ball = parse_raw_atd(raw_path, len(team), n_frames)
    print("  parsed {} frames x {} slots in {:.0f} s".format(n_frames, len(team), time.time() - t0))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, players=players, ball=ball,
             raw_bytes=np.int64(fp["bytes"]), raw_mtime=np.int64(fp["mtime"]))
    return team, names, players, ball


# -- small numeric helpers (explicit NaN handling, no warning suppression) -----------------------

def masked_mean(values, axis):
    """Mean over non-NaN entries; NaN where there are none."""
    ok = ~np.isnan(values)
    count = ok.sum(axis=axis)
    total = np.where(ok, values, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full(count.shape, np.nan, dtype=np.float64),
                     where=count > 0), count


def masked_min(values, axis):
    filled = np.where(np.isnan(values), np.inf, values)
    out = filled.min(axis=axis)
    return np.where(np.isinf(out), np.nan, out)


def central_diff(x, k, fps):
    """(x[t+k] - x[t-k]) / (2k / fps). NaN at the array edges and wherever an end is NaN."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    out[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2.0 * k / fps)
    return out


def run_lengths_of_missing(seen):
    """For each frame: length of the missing run it sits in (0 if seen), frames since the ball was
    last seen and until it is next seen (NaN if never)."""
    n = len(seen)
    idx = np.arange(n)
    last = np.where(seen, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(seen, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    since = np.where(last >= 0, idx - last, np.nan).astype(np.float64)
    until = np.where(nxt < n, nxt - idx, np.nan).astype(np.float64)
    gap = np.where(seen, 0.0, since + until - 1)  # NaN at an unbounded edge
    return gap, since, until


def interpolate_gaps(values, seen, max_gap, blocks):
    """Linear fill of missing runs of length <= max_gap whose both ends are seen, inside blocks.

    blocks: [(start, stop)] half-open spans; no run is filled across a block edge.
    """
    out = values.astype(np.float64).copy()
    filled = np.zeros(len(values), dtype=bool)
    for start, stop in blocks:
        s = seen[start:stop]
        if s.sum() < 2:
            continue
        missing = (~s).astype(np.int8)
        edges = np.diff(np.concatenate(([0], missing, [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if a == 0 or b == len(s) or (b - a) > max_gap:
                continue  # unbounded or too long: leave NaN, an honest gap
            lo, hi = start + a - 1, start + b
            w = (np.arange(start + a, start + b) - lo) / float(hi - lo)
            out[start + a:start + b] = out[lo] + w * (out[hi] - out[lo])
            filled[start + a:start + b] = True
    return out, filled


def block_spans(game, n_frames):
    """Half-open spans [pre, 1st half, halftime, 2nd half, post]: interpolation never crosses."""
    halves = half_video_blocks(game)
    cuts = [0]
    for _, first, last in halves:
        cuts += [first, last + 1]
    cuts.append(n_frames)
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


# -- feature blocks -----------------------------------------------------------------------------

def ball_features(ball_raw, pitch, blocks, cfg):
    length, width = pitch
    fps, k = cfg["fps"], cfg["velocity_half_window_frames"]
    seen = ~np.isnan(ball_raw[:, 0]) & ~np.isnan(ball_raw[:, 1])
    x_raw = ball_raw[:, 0].astype(np.float64) * length
    y_raw = ball_raw[:, 1].astype(np.float64) * width
    x, filled = interpolate_gaps(x_raw, seen, cfg["ball_max_interp_gap_frames"], blocks)
    y, _ = interpolate_gaps(y_raw, seen, cfg["ball_max_interp_gap_frames"], blocks)
    gap, since, until = run_lengths_of_missing(seen)

    vx, vy = central_diff(x, k, fps), central_diff(y, k, fps)
    speed = np.hypot(vx, vy)
    ax, ay = central_diff(vx, k, fps), central_diff(vy, k, fps)

    # Turn angle between the incoming (t-k -> t) and outgoing (t -> t+k) displacement, degrees.
    # Only defined when both legs move faster than turn_min_speed: a near-still ball has no
    # meaningful direction.
    inx = np.full(len(x), np.nan); iny = inx.copy(); outx = inx.copy(); outy = inx.copy()
    inx[k:] = x[k:] - x[:-k]; iny[k:] = y[k:] - y[:-k]
    outx[:-k] = x[k:] - x[:-k]; outy[:-k] = y[k:] - y[:-k]
    min_disp = cfg["turn_min_speed_m_s"] * k / fps
    ok = (np.hypot(inx, iny) > min_disp) & (np.hypot(outx, outy) > min_disp)
    cos = np.full(len(x), np.nan)
    cos[ok] = ((inx * outx + iny * outy)[ok]
               / (np.hypot(inx, iny) * np.hypot(outx, outy))[ok])
    turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    # Stillness: share of frames in the window whose speed is known and below threshold.
    h = cfg["ball_still_half_window_frames"]
    still = (speed < cfg["ball_still_speed_m_s"]).astype(np.float64)
    known = (~np.isnan(speed)).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(still))); ck = np.concatenate(([0.0], np.cumsum(known)))
    i = np.arange(len(x)); lo = np.maximum(0, i - h); hi = np.minimum(len(x), i + h + 1)
    n_known = ck[hi] - ck[lo]
    still_frac = np.divide(cs[hi] - cs[lo], n_known, out=np.full(len(x), np.nan),
                           where=n_known > 0)

    # Signed distance to the pitch boundary: > 0 outside, < 0 inside (distance to nearest line).
    dx_out = np.maximum(-x, x - length); dy_out = np.maximum(-y, y - width)
    outside = np.hypot(np.maximum(dx_out, 0), np.maximum(dy_out, 0))
    inside = np.maximum(dx_out, dy_out)  # negative when inside
    out_m = np.where((dx_out > 0) | (dy_out > 0), outside, inside)

    feats = {
        "ball_seen": seen, "ball_interp": filled,
        "ball_x_m": x, "ball_y_m": y,
        "ball_gap_len": gap, "ball_since_seen": since, "ball_until_seen": until,
        "ball_vx": vx, "ball_vy": vy, "ball_speed": speed,
        "ball_accel": np.hypot(ax, ay), "ball_turn_deg": turn,
        "ball_still_frac": still_frac, "ball_out_m": out_m,
    }
    for f in range(2, ball_raw.shape[1]):  # meaning not documented -> kept raw, unnamed
        feats["ball_raw_f{}".format(f)] = ball_raw[:, f].astype(np.float64)
    return feats


def attack_signs(px_m, team, cfg):
    """{half: +1/-1 for HOME attacking toward +x}. Read off the kickoff formation."""
    signs = {}
    w = cfg["kickoff_window_frames"]
    for half in (1, 2):
        k = cfg["kickoff_video_frames"][str(half)]
        window = px_m[k - w:k + 1]
        home_x, _ = masked_mean(window[:, team == HOME].reshape(-1), axis=0)
        away_x, _ = masked_mean(window[:, team == AWAY].reshape(-1), axis=0)
        sep = float(away_x - home_x)
        assert abs(sep) >= cfg["kickoff_min_centroid_separation_m"], (
            "half {}: team centroids only {:.1f} m apart along x at kickoff -- orientation or "
            "kickoff frame is wrong".format(half, sep))
        signs[half] = 1 if sep > 0 else -1
    return signs


def team_and_cross_features(players, team, ball_x, ball_y, half, signs, pitch, cfg):
    length, width = pitch
    fps, k = cfg["fps"], cfg["velocity_half_window_frames"]
    px = players[:, :, 0].astype(np.float64) * length
    py = players[:, :, 1].astype(np.float64) * width
    ps = players[:, :, 2].astype(np.float64)

    # Per-slot velocity; jumps faster than any human (identity switches) are dropped to NaN.
    vx = (np.full(px.shape, np.nan))
    vx[k:-k] = (px[2 * k:] - px[:-2 * k]) / (2.0 * k / fps)
    vy = np.full(py.shape, np.nan)
    vy[k:-k] = (py[2 * k:] - py[:-2 * k]) / (2.0 * k / fps)
    implausible = np.hypot(vx, vy) > cfg["player_max_plausible_speed_m_s"]
    vx[implausible] = np.nan; vy[implausible] = np.nan
    speed_check = speed_unit_ratio(ps, vx, vy, half, k, cfg["speed_unit_check_moving_m_s"])

    # Attack direction per frame for HOME; away is the opposite. 0 outside the halves.
    home_sign = np.zeros(len(px))
    home_sign[half == 1] = signs[1]; home_sign[half == 2] = signs[2]

    dist = np.hypot(px - ball_x[:, None], py - ball_y[:, None])  # NaN if ball or player missing
    centre_x = length / 2.0
    feats = {}
    dmins = {}
    for code, prefix in ((HOME, "h_"), (AWAY, "a_")):
        m = team == code
        sign = home_sign if code == HOME else -home_sign
        cx, n = masked_mean(px[:, m], axis=1)
        cy, _ = masked_mean(py[:, m], axis=1)
        spread, _ = masked_mean(np.hypot(px[:, m] - cx[:, None], py[:, m] - cy[:, None]), axis=1)
        sx, _ = masked_mean((px[:, m] - cx[:, None]) ** 2, axis=1)
        sy, _ = masked_mean((py[:, m] - cy[:, None]) ** 2, axis=1)
        mean_speed, _ = masked_mean(ps[:, m], axis=1)
        walking, _ = masked_mean(np.where(np.isnan(ps[:, m]), np.nan,
                                          (ps[:, m] < cfg["player_walking_speed_m_s"]) * 1.0), 1)
        mvx, _ = masked_mean(vx[:, m], axis=1)
        own_half, _ = masked_mean(np.where(np.isnan(px[:, m]), np.nan,
                                           (sign[:, None] * (px[:, m] - centre_x) < 0) * 1.0), 1)
        d = dist[:, m]
        dmin = masked_min(d, axis=1)
        d_sorted = np.sort(np.where(np.isnan(d), np.inf, d), axis=1)
        d2 = np.where(np.isinf(d_sorted[:, 1]), np.nan, d_sorted[:, 1])
        dmins[code] = dmin
        feats.update({
            prefix + "n_tracked": n.astype(np.int16),
            prefix + "cx_m": cx, prefix + "cy_m": cy,
            prefix + "std_x_m": np.sqrt(sx), prefix + "std_y_m": np.sqrt(sy),
            prefix + "spread_m": spread,
            prefix + "mean_speed": mean_speed, prefix + "walking_frac": walking,
            # position/velocity along this team's attacking direction (+ = toward opponent goal)
            prefix + "adv_x_m": np.where(sign != 0, sign * (cx - centre_x), np.nan),
            prefix + "adv_vx": np.where(sign != 0, sign * mvx, np.nan),
            prefix + "own_half_frac": np.where(sign != 0, own_half, np.nan),
            prefix + "dmin_ball_m": dmin, prefix + "d2_ball_m": d2,
        })
        for r in cfg["near_ball_radii_m"]:
            feats["{}n_within_{:g}m".format(prefix, r)] = np.where(
                np.isnan(ball_x), np.nan, (d < r).sum(axis=1))

    # Nearest player overall, and how much room they have (distance to nearest opponent).
    filled = np.where(np.isnan(dist), np.inf, dist)
    nearest_slot = filled.argmin(axis=1)
    has = np.isfinite(filled[np.arange(len(filled)), nearest_slot])
    nearest_team = np.where(has, team[nearest_slot], -1).astype(np.int8)
    nx = px[np.arange(len(px)), nearest_slot]; ny = py[np.arange(len(py)), nearest_slot]
    opp = team[None, :] != nearest_team[:, None]
    d_opp = np.where(opp, np.hypot(px - nx[:, None], py - ny[:, None]), np.nan)
    free = np.where(has, masked_min(d_opp, axis=1), np.nan)

    all_speed, _ = masked_mean(ps, axis=1)
    # Tracked = has a position. Speed can be missing on a tracked slot (e.g. a track's first
    # frame), so counting speeds would undercount; that shortfall is reported separately.
    n_all = (~np.isnan(px)).sum(axis=1)
    n_no_speed = (~np.isnan(px) & np.isnan(ps)).sum(axis=1)
    all_walk, _ = masked_mean(np.where(np.isnan(ps), np.nan,
                                       (ps < cfg["player_walking_speed_m_s"]) * 1.0), axis=1)
    feats.update({
        "nearest_team": nearest_team,  # 0 home, 1 away, -1 no ball or no players
        "nearest_margin_m": dmins[AWAY] - dmins[HOME],  # > 0: home closer
        "nearest_free_space_m": free,
        "all_mean_speed": all_speed, "all_walking_frac": all_walk,
        "n_tracked": n_all.astype(np.int16),
        "n_tracked_no_speed": n_no_speed.astype(np.int16),
        "n_implausible_velocity": implausible.sum(axis=1).astype(np.int16),
        "ball_dist_centre_m": np.hypot(ball_x - centre_x, ball_y - width / 2.0),
    })
    return feats, speed_check


def speed_unit_ratio(ps, vx, vy, half, k, moving_m_s):
    """Median of raw speed field / position speed over moving player-frames. Label-free unit check.

    ps, vx, vy [N, S]; position speed is the per-slot central difference (implausible jumps
    already NaN). A sample counts only if t-k, t and t+k are all in the same graded half, both
    speeds are finite and the position speed exceeds moving_m_s. Returns (ratio or NaN, n).
    """
    assert ps.shape == vx.shape == vy.shape and ps.shape[0] == len(half)
    pos_speed = np.hypot(vx, vy)
    same = np.zeros(len(half), dtype=bool)
    same[k:-k] = (half[k:-k] > 0) & (half[:-2 * k] == half[k:-k]) & (half[2 * k:] == half[k:-k])
    use = same[:, None] & np.isfinite(ps) & np.isfinite(pos_speed) & (pos_speed > moving_m_s)
    n = int(use.sum())
    ratio = float(np.median(ps[use] / pos_speed[use])) if n else float("nan")
    return {"ratio": ratio, "n_samples": n}


def smart_tagging_features(path, n_frames, cfg):
    """One bool column per (code, team) active on the frame. Intervals are [start, end)."""
    team_map = cfg["smart_tagging_team_map"]
    fps = cfg["fps"]
    cols = {}
    for inst in ET.parse(path).iter("instance"):
        code = inst.findtext("code").strip().lower().replace(" ", "_")
        tags = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}
        team_raw = tags.get("Team")
        assert team_raw in team_map, "Smart Tagging Team {!r} not in config team map".format(
            team_raw)
        start = int(round(float(inst.findtext("start")) * fps))
        end = int(round(float(inst.findtext("end")) * fps))
        assert 0 <= start <= end <= n_frames, "tag {} [{}, {}) outside the video".format(
            code, start, end)
        name = "st_{}_{}".format(code, team_map[team_raw])
        if name not in cols:
            cols[name] = np.zeros(n_frames, dtype=bool)
        cols[name][start:end] = True
    return dict(sorted(cols.items()))


# -- config provenance ---------------------------------------------------------------------------

PROVENANCE_BASES = {"published", "law_of_game", "physics", "a_priori", "tuned_external"}
PROVENANCE_FIELDS = {"basis", "source", "reason", "tuned_on"}


def check_provenance(cfg):
    """Every config value must say where it came from (organisers' rules, 2026-11-03).

    Fails if a key has no provenance entry, an entry has no matching key, a field is missing, the
    basis is not one of PROVENANCE_BASES, or tuned_on disagrees with basis. 'tuned_external' is
    the only basis that may (and must) name a tuning dataset, and it must not be the provided game.
    """
    keys = {k for k in cfg if not k.startswith("_")}
    prov = cfg["_provenance"]
    missing, orphan = keys - set(prov), set(prov) - keys
    assert not missing, "config keys without provenance: {}".format(sorted(missing))
    assert not orphan, "provenance for keys not in config: {}".format(sorted(orphan))
    for key, entry in prov.items():
        absent = PROVENANCE_FIELDS - set(entry)
        assert not absent, "provenance of {!r} lacks {}".format(key, sorted(absent))
        assert entry["basis"] in PROVENANCE_BASES, "provenance of {!r}: basis {!r} not in {}".format(
            key, entry["basis"], sorted(PROVENANCE_BASES))
        tuned = entry["tuned_on"]
        if entry["basis"] == "tuned_external":
            assert tuned, "{!r} is tuned_external but names no dataset".format(key)
            assert "BAR-ATH" not in str(tuned), \
                "{!r} claims tuning on the provided game -- not allowed".format(key)
        else:
            assert tuned is None, "{!r} has tuned_on but basis {!r}".format(key, entry["basis"])


# -- driver -------------------------------------------------------------------------------------

def half_index(game, n_frames):
    half = np.zeros(n_frames, dtype=np.int8)
    for h, first, last in half_video_blocks(game):
        half[first:last + 1] = h
    return half


def build(game, data_dir, cfg, use_cache):
    check_provenance(cfg)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    assert cfg["fps"] == GAMES[game]["fps"], "config fps != games.py fps"
    cache = os.path.join(data_dir or DATA_DIR, "derived", "atd_tactical_cache.npz")

    print("loading tracking ...")
    team, names, players, ball_raw = load_arrays(meta_path, raw_path, n_frames, cache, use_cache)
    print("  H = {}, A = {}; {} slots ({} home, {} away); ball has {} raw fields".format(
        names[0], names[1], len(team), int((team == HOME).sum()), int((team == AWAY).sum()),
        ball_raw.shape[1]))
    speeds = players[:, :, 2]
    assert np.nanmax(speeds) < 3 * cfg["player_max_plausible_speed_m_s"], \
        "player speed field max {:.1f} -- not m/s?".format(np.nanmax(speeds))

    half = half_index(game, n_frames)
    blocks = block_spans(game, n_frames)
    print("computing features ...")
    feats = {"frame": np.arange(n_frames, dtype=np.int32), "half": half, "graded": half > 0}
    feats.update(ball_features(ball_raw, pitch, blocks, cfg))
    signs = attack_signs(players[:, :, 0].astype(np.float64) * pitch[0], team, cfg)
    print("  home attacks {} x in half 1, {} x in half 2".format(
        "+" if signs[1] > 0 else "-", "+" if signs[2] > 0 else "-"))
    assert signs[1] == -signs[2], "teams did not switch ends at halftime -- check kickoff frames"
    team_feats, speed_check = team_and_cross_features(players, team, feats["ball_x_m"],
                                                      feats["ball_y_m"], half, signs, pitch, cfg)
    feats.update(team_feats)

    st_path = require(["smart_tagging"], game, data_dir)[0]
    feats.update(smart_tagging_features(st_path, n_frames, cfg))

    lengths = {k: len(v) for k, v in feats.items()}
    assert set(lengths.values()) == {n_frames}, "column length mismatch: {}".format(lengths)
    df = pd.DataFrame(feats)
    # float64 is used for the arithmetic; float32 is plenty for storage (sub-mm in metres).
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype(np.float32)
    inputs = [file_fingerprint(p) for p in (meta_path, raw_path, st_path)]
    return df, {"attack_sign_home": signs, "teams": names, "inputs": inputs,
                "speed_unit_check": speed_check}


def provenance(cfg, info, out_path):
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):  # not a git checkout: record that
            return None
    with open(os.path.join(HERE, "extract_features.py"), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    return {
        "output": os.path.abspath(out_path), "config": cfg, **info,
        "script_sha256": script_sha, "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(sh(["git", "status", "--porcelain"])),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    }


def speed_unit_verdict(ratio, tol):
    """Interpretation only, never used for conversion: m/s, km/h or unknown."""
    for unit, ref in (("m/s", 1.0), ("km/h", 3.6)):
        if abs(ratio / ref - 1) <= tol:
            return unit
    return "unknown"


def report_summary(df, info, cfg):
    """Label-free sanity summary over the graded halves.

    Deliberately reads NO ground truth, public or private: under the organisers' 2026-11-03
    clarification nothing may be fitted or calibrated on the GT, so this report only checks that
    the features are physically plausible. The DATA.md column gives the published figure to
    compare against where one exists.
    """
    g = df[df["graded"]]
    print("\nsanity summary, graded frames only ({} frames)".format(len(g)))
    rows = [
        ("ball seen (raw)", g["ball_seen"].mean(), "59.3% (DATA.md)"),
        ("ball seen or interpolated", (g["ball_seen"] | g["ball_interp"]).mean(), "> raw"),
        ("mean tracked objects", g["n_tracked"].mean(), "20.5 (DATA.md)"),
        ("tracked slots with no speed (mean)", g["n_tracked_no_speed"].mean(), "~0"),
        # share of frames that HAVE a nearest player (ball position known); the two sum to 100%
        ("nearest player is home | ball known",
         (g["nearest_team"] == 0).sum() / max(1, (g["nearest_team"] >= 0).sum()), "30-70%"),
        ("nearest player is away | ball known",
         (g["nearest_team"] == 1).sum() / max(1, (g["nearest_team"] >= 0).sum()), "30-70%"),
        ("ball outside pitch", (g["ball_out_m"] > 0).mean(), "small"),
        ("ball median speed m/s", g["ball_speed"].median(), "~3-10"),
        ("player median speed m/s", g["all_mean_speed"].median(), "~1-3"),
        ("ball p99 speed m/s", g["ball_speed"].quantile(0.99), "< ~40"),
        ("frames w/ implausible track velocity", (g["n_implausible_velocity"] > 0).mean(),
         "small"),
    ]
    for name, value, expect in rows:
        shown = "{:.1%}".format(value) if expect.endswith("%") or name.startswith(
            ("ball seen", "nearest", "ball outside", "frames")) else "{:.2f}".format(value)
        print("  {:<40} {:>8}   expect {}".format(name, shown, expect))
    # Ball gaps: how many missing runs inside the graded halves are short enough to interpolate.
    lengths = []
    for _, half_rows in g.groupby("half"):  # per half, so no run joins across halftime
        missing = (~half_rows["ball_seen"].values).astype(np.int8)
        edges = np.diff(np.concatenate(([0], missing, [0])))
        lengths.append(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1))
    lengths = np.concatenate(lengths)
    print("  ball-missing runs in graded frames: {}".format(len(lengths)))
    for lo, hi in ((1, 12), (13, 25), (26, 125), (126, 10 ** 9)):
        sel = (lengths >= lo) & (lengths <= hi)
        print("    {:>4}-{:<5} frames: {:>6} runs, {:>6.1%} of missing frames".format(
            lo, hi if hi < 10 ** 9 else "inf", int(sel.sum()),
            lengths[sel].sum() / max(1, lengths.sum())))
    print("  frames filled by interpolation: {}".format(int(g["ball_interp"].sum())))
    extra = [c for c in df.columns if c.startswith("ball_raw_f")]
    print("  extra raw ball fields: {}".format(extra or "none"))
    for c in extra:
        v = g[c].dropna()
        print("    {}: {} distinct, min {:.4f}, median {:.4f}, p99 {:.4f}, max {:.4f}".format(
            c, v.nunique(), v.min(), v.median(), v.quantile(0.99), v.max()))
    st = [c for c in df.columns if c.startswith("st_")]
    print("  smart tagging columns: {}".format(len(st)))
    # Unit of the raw per-player speed field (players[:, :, 2]): compare with position speed.
    sc = info["speed_unit_check"]
    print("  raw player speed / position speed (median, moving > {:g} m/s, within a half): "
          "{:.3f} over {} player-frames".format(cfg["speed_unit_check_moving_m_s"], sc["ratio"],
                                                sc["n_samples"]))
    print("    interpretation (~1 = m/s, ~3.6 = km/h, else unknown; tol {:.0%}): {}".format(
        cfg["speed_unit_check_tolerance"],
        speed_unit_verdict(sc["ratio"], cfg["speed_unit_check_tolerance"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(HERE, "config.json"))
    parser.add_argument("--data-dir", default=None, help="defaults to MTAH_DATA_DIR or ./data")
    parser.add_argument("--out", default=None,
                        help="default <data-dir>/derived/features_tactical.parquet")
    parser.add_argument("--no-cache", action="store_true", help="re-parse the raw ATD")
    parser.add_argument("--report", action="store_true",
                        help="print a label-free sanity summary (reads no ground truth)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    out = args.out or os.path.join(args.data_dir or DATA_DIR, "derived",
                                   "features_tactical.parquet")
    df, info = build(GAME, args.data_dir, cfg, use_cache=not args.no_cache)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_parquet(out, index=False)
    with open(os.path.splitext(out)[0] + "_provenance.json", "w") as f:
        json.dump(provenance(cfg, info, out), f, indent=2, default=str)
    print("\nwrote {}  ({} rows x {} cols, {:.1f} MB)".format(
        out, len(df), df.shape[1], os.path.getsize(out) / 1e6))
    if args.report:
        report_summary(df, info, cfg)


if __name__ == "__main__":
    main()
