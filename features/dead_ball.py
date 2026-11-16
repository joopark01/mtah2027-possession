"""Rule-based dead-ball detector v1 (docs/dead-ball-detector-design.md).

    out = detect(df, cfg, ctx)        # {"dead", "segments", "restart_team"}
    labels = predict_labels(df, cfg, ctx)

df: a feature table (extract_features.py or idsse_features.py; `label` and `ball_dropped` must
    not be in it).
cfg: features/config.json, the dead_* keys (ball_still_speed_m_s enters only through the table's
     ball_still_frac). No defaults here.
ctx: per-game context, never hard-coded:
    known_dead     [(start, stop)] half-open published dead spans (BAR-ATH: [(110808, 113742)];
                   IDSSE: []). Each must sit inside one `half` block.
    smart_tagging  True if the table has st_ columns (BAR-ATH; at least one st_set_pieces_*
                   column required), False if it has none (IDSSE). Checked.

Everything is per `half` block (ef.block_spans): on IDSSE the two periods are adjacent rows, so
no smoothing, run, merge or segment crosses a block edge. Intervals are half-open [start, stop).

Candidates (design section 2): (a) long ball gaps where the players have stopped, (b) the ball
seen out of play, (c) a placed ball (seen, still, players stopped), (d) ctx known_dead. A long gap
where the players keep running is not D (2.2). D candidates separated by an alive blip shorter than
dead_min_alive_frames are merged (2.3). After merging, every D segment shorter than
dead_min_dead_frames becomes alive (0 = off); a segment holding any ctx known_dead frame is kept.

Boundaries (section 3), for a D gap [v, e) of the unknown ball: end = e + reappear offset (the
first frame the ball is seen again; reappearances are treated as moving, findings: only 2% are
still, and a still one is extended by candidate (c)). Start = the line crossing if the ball was
last seen out of play (candidate (b) supplies it), else the onset of the stopped run (smoothed
speed and walking) shifted back by the whistle->deceleration lag, else v + vanish offset.
The onset is the start of the stopped run containing v, or else the first stopped frame inside
the gap. In that second case start = onset - lag can fall after v (the design's order, taken
literally). The run containing v may reach back at most dead_onset_max_lookback_frames: if it
starts earlier, start = v - that bound, with no lag (the stretch before is taken as slow live
play). If it starts within the bound, the lag is applied as usual, so start can be up to bound + lag
before v. A reappearance whose ball speed is still NaN (the first k frames) counts as moving.

Further choices (accepted 2026-11-15): "walking share" = min of the two teams' shares, else
all_walking_frac where a team's share is NaN. A long gap is live if its median smoothed walking
share is below the threshold or its median smoothed speed above it; a gap with no finite player
data is unknown and is not called live.

Restart team (section 4): set-piece tag team (H/A; N/A falls through), goal tag (the conceding
team), nearest player when play restarts, else "unknown". Goal-line geometry (goal kick vs corner)
is deferred to v2: it needs a last-touch proxy. Smart Tagging meanings are assumptions, not
documented facts: the SET PIECES Team is the team taking the restart, the GOALS Team the team that
scored. A set-piece tag counts if it overlaps [start, end], the restart frame included.
"""

import numpy as np
import pandas as pd

import extract_features as ef

# Keys the IDSSE tuning step may grid. Everything else is fixed a priori (the vanish/reappear
# offsets are never tuned: on degraded IDSSE they are set by the simulator's assumptions).
TUNABLE = ("dead_decel_lag_frames", "dead_smooth_half_window_frames", "dead_walking_frac_min",
           "dead_mean_speed_max_m_s", "dead_min_gap_frames", "dead_min_alive_frames",
           "dead_min_dead_frames")
CTX_KEYS = {"known_dead", "smart_tagging"}
TEAM = np.array(["H", "A"])


def _runs(mask, blocks):
    """Half-open (start, stop) runs of True, never crossing a block edge."""
    out = []
    for a, b in blocks:
        d = np.diff(np.concatenate(([0], mask[a:b].astype(np.int8), [0])))
        out += [(a + s, a + t) for s, t in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return out


def _block_mean(x, h, blocks):
    """NaN-aware centred mean over [t-h, t+h] clipped to t's block; NaN where nothing is known."""
    n = len(x)
    ok = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    ck = np.concatenate(([0], np.cumsum(ok)))
    lo = np.empty(n, np.int64); hi = np.empty(n, np.int64)
    i = np.arange(n)
    for a, b in blocks:
        lo[a:b] = np.maximum(a, i[a:b] - h); hi[a:b] = np.minimum(b, i[a:b] + h + 1)
    k = ck[hi] - ck[lo]
    return np.divide(cs[hi] - cs[lo], k, out=np.full(n, np.nan), where=k > 0)


def _col(df, name):
    return df[name].to_numpy() if name in df else np.zeros(len(df), bool)


def detect(df, cfg, ctx):
    assert set(ctx) == CTX_KEYS, "ctx keys {} != {}".format(sorted(ctx), sorted(CTX_KEYS))
    for leak in ("label", "ball_dropped"):
        if leak in df:
            raise ValueError("detector must not see {!r}".format(leak))
    for k in ("dead_decel_lag_frames", "dead_vanish_offset_frames", "dead_reappear_offset_frames",
              "dead_onset_max_lookback_frames"):
        if cfg[k] < 0:
            raise ValueError("{} = {} is negative".format(k, cfg[k]))
    n = len(df)
    has_st = any(c.startswith("st_") for c in df.columns)
    assert has_st == ctx["smart_tagging"], "ctx smart_tagging={} but st_ columns {}".format(
        ctx["smart_tagging"], "present" if has_st else "absent")
    if ctx["smart_tagging"] and not any(c.startswith("st_set_pieces_") for c in df.columns):
        raise ValueError("ctx smart_tagging=True but no st_set_pieces_* column")
    half = df["half"].to_numpy()
    blocks = ef.block_spans(half)
    bid = ef.block_ids(blocks, n)
    bstop = np.array([b for _, b in blocks])[bid]
    h = cfg["dead_smooth_half_window_frames"]
    assert h >= 0 and cfg["dead_min_gap_frames"] >= 1 and cfg["dead_min_alive_frames"] >= 0
    assert cfg["dead_min_dead_frames"] >= 0, "dead_min_dead_frames negative"

    # player state, smoothed; NaN = unknown (findings 2026-11-15: NaN team walking share falls
    # back to all_walking_frac, never read as "not walking")
    hw, aw = df["h_walking_frac"].to_numpy(float), df["a_walking_frac"].to_numpy(float)
    walk = np.where(np.isfinite(hw) & np.isfinite(aw), np.fmin(hw, aw),
                    df["all_walking_frac"].to_numpy(float))
    walk = _block_mean(walk, h, blocks)
    speed = _block_mean(df["all_mean_speed"].to_numpy(float), h, blocks)
    walking_ok = walk >= cfg["dead_walking_frac_min"]           # False where NaN
    slow_ok = speed <= cfg["dead_mean_speed_max_m_s"]
    stopped = walking_ok & slow_ok
    prev = np.concatenate(([False], stopped[:-1] & (bid[:-1] == bid[1:])))
    run_start = np.where(stopped & ~prev, np.arange(n), -1)
    run_start = np.maximum.accumulate(run_start)                # start of the stopped run at t

    known = df["ball_x_m"].notna().to_numpy()
    out_ = known & (df["ball_out_m"].to_numpy(float) > cfg["dead_out_margin_m"])
    still = df["ball_still_frac"].to_numpy(float) >= cfg["dead_placed_still_frac_min"]
    dead = out_ | (known & still & stopped)                     # (b), (c)

    for v, e in _runs(~known, blocks):                          # (a) long gaps
        if e - v < cfg["dead_min_gap_frames"]:
            continue
        # running = known evidence that players are NOT stopped over the gap (median); a feature
        # with no finite value in the gap is unknown and does not make the gap live
        w, s = walk[v:e], speed[v:e]
        if (np.isfinite(w).any() and np.median(w[np.isfinite(w)]) < cfg["dead_walking_frac_min"]) \
                or (np.isfinite(s).any()
                    and np.median(s[np.isfinite(s)]) > cfg["dead_mean_speed_max_m_s"]):
            continue
        a = blocks[bid[v]][0]
        if v > a and out_[v - 1]:
            start = v                                           # crossing: (b) supplies it
        elif stopped[v]:
            o, cap = run_start[v], cfg["dead_onset_max_lookback_frames"]
            start = v - cap if v - o > cap else o - cfg["dead_decel_lag_frames"]
        elif stopped[v:e].any():
            start = v + int(np.argmax(stopped[v:e])) - cfg["dead_decel_lag_frames"]
        else:
            start = v + cfg["dead_vanish_offset_frames"]
        stop = min(e + cfg["dead_reappear_offset_frames"], bstop[v])
        dead[max(start, a):stop] = True

    kd = np.zeros(n, bool)
    for a, b in ctx["known_dead"]:                              # (d)
        assert 0 <= a < b <= n and bid[a] == bid[b - 1], "known_dead {} not inside one block".format((a, b))
        kd[a:b] = True
    dead |= kd

    for a, b in _runs(~dead, blocks):                           # merge short alive blips
        if b - a < cfg["dead_min_alive_frames"] and a > blocks[bid[a]][0] and b < bstop[a]:
            dead[a:b] = True

    for a, b in _runs(dead, blocks):                            # drop short D segments
        if b - a < cfg["dead_min_dead_frames"] and not kd[a:b].any():
            dead[a:b] = False

    segments = _runs(dead, blocks)
    return {"dead": dead, "segments": segments,
            "restart_team": [_restart_team(df, s, e, bstop[s]) for s, e in segments]}


def _restart_team(df, s, e, block_stop):
    hi = min(e + 1, block_stop)                                 # the restart frame e included
    sp = [t for t in "HA" if _col(df, "st_set_pieces_" + t)[s:hi].any()]
    if len(sp) == 1:
        return sp[0]
    goal = [t for t in "HA" if _col(df, "st_goals_" + t)[s:e].any()]
    if len(goal) == 1:
        return "A" if goal[0] == "H" else "H"                   # the conceding team kicks off
    if e < block_stop:
        nt = int(df["nearest_team"].iat[e])
        if nt in (0, 1) and np.isfinite(df["ball_x_m"].iat[e]):
            return str(TEAM[nt])
    return "unknown"


def predict_labels(df, cfg, ctx):
    """D on dead frames; alive frames: nearest-player team where the ball is known, else the last
    alive team carried forward within the block, else the next one, else D."""
    dead = detect(df, cfg, ctx)["dead"]
    nt = df["nearest_team"].to_numpy()
    known = df["ball_x_m"].notna().to_numpy()
    team = pd.Series(np.where(~dead & known & (nt >= 0), nt, np.nan))
    bid = pd.Series(ef.block_ids(ef.block_spans(df["half"].to_numpy()), len(df)))
    team = team.groupby(bid).ffill().groupby(bid).bfill().to_numpy()
    lab = np.where(dead | np.isnan(team), "D", TEAM[np.nan_to_num(team).astype(int)])
    assert len(lab) == len(df) and set(np.unique(lab)) <= {"H", "A", "D"}
    return lab
