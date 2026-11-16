"""Possession decoder v1: per-frame H/A/D/X from the feature table.

    labels = decode(df, cfg, ctx)            # np.array of 'H'/'A'/'D'/'X', one per row

df, ctx: as dead_ball.detect (ctx = {"known_dead", "smart_tagging"}). cfg: features/config.json;
the decoder_* keys plus the dead_* keys the detector reads. No defaults here.

1. dead_ball.detect: its dead frames are D (hard constraint). Ungraded rows (half 0) are X.
2. Each alive span (a run of non-dead rows inside one graded ef.block_spans block, so never across
   a half edge or the IDSSE period seam) is decoded on its own by Viterbi over C (contested /
   loose), H, A. Scores are log-likelihood-like, per frame:
     H = clip(nearest_margin_m / decoder_margin_full_m, -1, 1), A = -H   (ball position known)
     C = decoder_c_score where both teams or neither team has a player within 3 m, else 0
   With the ball position unknown (ball_x_m NaN) or the margin NaN, all three are 0 (flat), so
   the transition penalties carry the state through. Transitions: H<->A costs
   decoder_switch_penalty, H/A->C decoder_c_entry_penalty, C->H/A decoder_c_exit_penalty.
   Initial state: if the span starts at a detector D segment's end with restart_team H or A, the
   other two states start at -decoder_restart_bias; else all start at 0.
3. Output: a C run takes the label of the preceding H/A run in the same alive span, else the
   following one, so a loose ball never splits a spell. A span decoded as C throughout takes its
   majority nearest_team call (ball known), else the last H/A label before it in the block, else
   the next one after it, else D.

Ties (exactly equal scores) go to the lowest state index: C, then H, then A, so flat evidence
never invents a team. Deviation from the brief, flagged: C is resolved within the alive span, not
across D segments, because a spell never crosses D anyway and the start of a new span belongs to
the possession after the restart.
"""

import numpy as np
import pandas as pd

import dead_ball as db
import extract_features as ef

KEYS = ("decoder_margin_full_m", "decoder_c_score", "decoder_switch_penalty",
        "decoder_c_entry_penalty", "decoder_c_exit_penalty", "decoder_restart_bias")
C, H, A = 0, 1, 2
OUT = np.array(["C", "H", "A"])


def transition_matrix(cfg):
    """[from, to] log-scores."""
    s, en, ex = cfg["decoder_switch_penalty"], cfg["decoder_c_entry_penalty"], cfg["decoder_c_exit_penalty"]
    return -np.array([[0.0, ex, ex], [en, 0.0, s], [en, s, 0.0]])


def emissions(df, cfg):
    """[N, 3] per-frame scores for C, H, A."""
    assert cfg["decoder_margin_full_m"] > 0 and cfg["decoder_c_score"] >= 0
    known = df["ball_x_m"].notna().to_numpy()
    m = df["nearest_margin_m"].to_numpy(float)
    team = np.where(known & np.isfinite(m), np.clip(m / cfg["decoder_margin_full_m"], -1, 1), 0.0)
    h3 = df["h_n_within_3m"].to_numpy(float) > 0       # NaN (ball unknown) -> False
    a3 = df["a_n_within_3m"].to_numpy(float) > 0
    c = np.where(known & (h3 == a3), float(cfg["decoder_c_score"]), 0.0)
    return np.stack([c, team, -team], axis=1)


def viterbi(e, trans, init):
    """Most likely state path for scores e [T, 3], trans [3, 3], init [3]."""
    T = len(e)
    bp = np.zeros((T, 3), np.int8)
    v = init + e[0]
    # ponytail: Python loop over frames (~1 s per 200k); vectorise across spans if tuning is slow
    for t in range(1, T):
        cand = v[:, None] + trans
        bp[t] = cand.argmax(0)
        v = cand.max(0) + e[t]
    path = np.empty(T, np.int8)
    path[-1] = int(v.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def decode(df, cfg, ctx):
    return decode_from(df, cfg, db.detect(df, cfg, ctx))


def decode_from(df, cfg, det):
    """decode() with a precomputed dead_ball.detect result for this df (tuning reuses it)."""
    for k in KEYS:
        assert k in cfg, "config lacks " + k
    for k in ("decoder_switch_penalty", "decoder_c_entry_penalty", "decoder_c_exit_penalty",
              "decoder_restart_bias"):
        if cfg[k] < 0:
            raise ValueError("{} = {} is negative".format(k, cfg[k]))
    n = len(df)
    dead = det["dead"]
    assert dead.dtype == bool and len(dead) == n
    half = df["half"].to_numpy()
    blocks = [(a, b) for a, b in ef.block_spans(half) if half[a] > 0]
    restart = {e: t for (_, e), t in zip(det["segments"], det["restart_team"])}
    e_all, trans = emissions(df, cfg), transition_matrix(cfg)
    nt = df["nearest_team"].to_numpy()
    known = df["ball_x_m"].notna().to_numpy()

    lab = np.where(half > 0, "D", "X").astype("<U1")
    for ba, bb in blocks:
        pending = []                                    # all-C spans, resolved by block carry
        for s, t in db._runs(~dead, [(ba, bb)]):
            init = np.zeros(3)
            # a span at the block start has no D segment before it in this block (IDSSE seam)
            team = restart.get(s, "unknown") if s > ba else "unknown"
            if team in ("H", "A"):
                init[:] = -cfg["decoder_restart_bias"]
                init[H if team == "H" else A] = 0.0
            path = viterbi(e_all[s:t], trans, init)
            ha = path != C
            if ha.any():
                idx = np.where(ha, np.arange(t - s), -1)
                idx = np.maximum.accumulate(idx)
                idx[idx < 0] = np.flatnonzero(ha)[0]      # leading C -> following H/A
                lab[s:t] = OUT[path[idx]]
                continue
            calls = nt[s:t][known[s:t] & (nt[s:t] >= 0)]
            if len(calls) and (calls == 0).sum() != (calls == 1).sum():
                lab[s:t] = "H" if (calls == 0).sum() > (calls == 1).sum() else "A"
            else:
                pending.append((s, t))
        if pending:                                     # carry H/A within the block, never D
            seg = pd.Series(np.where(np.isin(lab[ba:bb], ["H", "A"]), lab[ba:bb], None))
            filled = seg.ffill().bfill()
            for s, t in pending:
                f = filled.iloc[s - ba]
                lab[s:t] = "D" if f is None or pd.isna(f) else f
    assert len(lab) == n and set(np.unique(lab)) <= {"H", "A", "D", "X"}
    assert ((lab == "X") == (half == 0)).all() and (lab[dead & (half > 0)] == "D").all()
    return lab
