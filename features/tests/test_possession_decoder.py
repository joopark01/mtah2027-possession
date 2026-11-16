"""Synthetic checks of features/possession_decoder.py, features/external/tune_decoder.py and
features/make_submission.py. Made-up tables only; make_submission runs on the synthetic match.

    python features/tests/make_synthetic_match.py /tmp/synth
    MTAH_DATA_DIR=/tmp/synth python features/tests/test_possession_decoder.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
_dir = os.environ.get("MTAH_DATA_DIR", "")
if not _dir or os.path.realpath(_dir) == os.path.realpath(os.path.join(ROOT, "data")):
    raise SystemExit("MTAH_DATA_DIR is empty or points at the real data/ dir; synthetic match only")
sys.path.insert(0, os.path.join(HERE, "..", "external"))
import tune_decoder as td  # noqa: E402

hz, dec = td.hz, td.pd_
with open(os.path.join(HERE, "..", "config.json")) as f:
    CFG = json.load(f)
# Test values, independent of later tuning: margin saturates at 2 m, switch costs 50.
CFG.update(decoder_margin_full_m=2.0, decoder_c_score=1.0, decoder_switch_penalty=50.0,
           decoder_c_entry_penalty=25.0, decoder_c_exit_penalty=25.0, decoder_restart_bias=50.0)
CTX = {"known_dead": [], "smart_tagging": False}


def table(n, halves=None, margin=3.0):
    """Live play: ball known in the pitch and moving, players running, home 3 m clearer, only home
    within 3 m."""
    return pd.DataFrame({
        "half": np.ones(n, np.int8) if halves is None else np.asarray(halves, np.int8),
        "ball_x_m": np.full(n, 50.0), "ball_out_m": np.full(n, -5.0), "ball_still_frac": np.zeros(n),
        "nearest_team": np.zeros(n, np.int8), "nearest_margin_m": np.full(n, margin),
        "h_n_within_3m": np.ones(n), "a_n_within_3m": np.zeros(n),
        "h_walking_frac": np.full(n, 0.3), "a_walking_frac": np.full(n, 0.3),
        "all_walking_frac": np.full(n, 0.3), "all_mean_speed": np.full(n, 3.0)})


def away(df, a, b, margin=-3.0):
    df.loc[a:b - 1, "nearest_margin_m"] = margin
    df.loc[a:b - 1, "nearest_team"] = 1
    df.loc[a:b - 1, ["h_n_within_3m", "a_n_within_3m"]] = [0.0, 1.0]


def det(n, segs, teams):
    dead = np.zeros(n, bool)
    for s, e in segs:
        dead[s:e] = True
    return {"dead": dead, "segments": segs, "restart_team": teams}


def runs(lab):
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    edges = np.concatenate(([0], b, [len(lab)]))
    return [(int(s), int(e), lab[s]) for s, e in zip(edges[:-1], edges[1:])]


def raises(exc, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc:
        return
    raise AssertionError("{} did not raise {}".format(fn.__name__, exc.__name__))


def main():
    # 1. A short away blip (10 frames, clear) inside a home spell does not split it.
    df = table(600)
    away(df, 300, 310)
    assert runs(dec.decode(df, CFG, CTX)) == [(0, 600, "H")]
    # ... a contested blip (both within 3 m, margin 0) neither.
    df = table(600)
    df.loc[300:339, ["nearest_margin_m", "a_n_within_3m"]] = [0.0, 1.0]
    assert runs(dec.decode(df, CFG, CTX)) == [(0, 600, "H")]

    # 2. Sustained change to away at 300: the boundary lands on the frame.
    df = table(600)
    away(df, 300, 600)
    assert runs(dec.decode(df, CFG, CTX)) == [(0, 300, "H"), (300, 600, "A")]
    # Contested 300..320 then away: the loose ball stays home's (C takes the preceding team).
    df = table(600)
    away(df, 320, 600)
    df.loc[300:319, ["nearest_margin_m", "h_n_within_3m", "a_n_within_3m"]] = [0.0, 1.0, 1.0]
    assert runs(dec.decode(df, CFG, CTX)) == [(0, 320, "H"), (320, 600, "A")]

    # 3. Ball unknown in live play (long gap, players running): flat, the team carries through.
    df = table(600)
    df.loc[200:399, ["ball_x_m", "ball_out_m", "ball_still_frac", "nearest_margin_m",
                     "h_n_within_3m", "a_n_within_3m"]] = np.nan
    df.loc[200:399, "nearest_team"] = -1
    assert runs(dec.decode(df, CFG, CTX)) == [(0, 600, "H")]

    # 4. Detector D stays D, even with strong home evidence; ball out 200..300 -> D exactly there.
    df = table(600)
    df.loc[200:299, "ball_out_m"] = 2.0
    assert runs(dec.decode(df, CFG, CTX)) == [(0, 200, "H"), (200, 300, "D"), (300, 600, "H")]

    # 5. X exactly on ungraded rows; nothing crosses a block edge (away at the end of half 1
    # does not leak into half 2).
    df = table(900, [0] * 100 + [1] * 300 + [0] * 100 + [2] * 400)
    away(df, 300, 400)
    lab = dec.decode(df, CFG, CTX)
    assert runs(lab) == [(0, 100, "X"), (100, 300, "H"), (300, 400, "A"), (400, 500, "X"),
                         (500, 900, "H")], runs(lab)

    # 6. Restart bias: weak home evidence after a stoppage whose restart team is A -> A; without
    # a known restart team -> H. Precomputed detector output via decode_from.
    df = table(400, margin=0.5)
    df.loc[0:99, "nearest_margin_m"] = 3.0
    df.loc[200:229, ["nearest_margin_m"]] = 0.5
    df.loc[230:, ["ball_x_m", "nearest_margin_m"]] = np.nan
    d = det(400, [(100, 200)], ["A"])
    assert runs(dec.decode_from(df, CFG, d)) == [(0, 100, "H"), (100, 200, "D"), (200, 400, "A")]
    d = det(400, [(100, 200)], ["unknown"])
    assert runs(dec.decode_from(df, CFG, d)) == [(0, 100, "H"), (100, 200, "D"), (200, 400, "H")]

    # 7. Span with no evidence and no restart team -> carried from the block; a block with
    # nothing known at all -> D.
    df = table(400)
    df.loc[200:, ["ball_x_m", "nearest_margin_m"]] = np.nan
    df.loc[200:, "nearest_team"] = -1
    away(df, 0, 100)
    lab = dec.decode_from(df, CFG, det(400, [(100, 200)], ["unknown"]))
    assert runs(lab) == [(0, 100, "A"), (100, 200, "D"), (200, 400, "A")], runs(lab)
    df.loc[:, ["ball_x_m", "nearest_margin_m"]] = np.nan
    lab = dec.decode_from(df, CFG, det(400, [(100, 200)], ["unknown"]))
    assert set(lab) == {"D"}, runs(lab)

    # 8. Noise: iid random margins never produce short spells. An inner H/A run costs at least 50
    # in transitions and gains at most 2 per frame, so no run inside a span is shorter than 25.
    rng = np.random.default_rng(0)
    df = table(20000)
    df["nearest_margin_m"] = rng.normal(0, 3, len(df))
    df["a_n_within_3m"] = rng.integers(0, 2, len(df)).astype(float)
    r = runs(dec.decode(df, CFG, CTX))
    inner = [e - s for s, e, _ in r[1:-1]]
    assert not inner or min(inner) >= 25, min(inner)

    # 9. Guards: negative penalties, missing key.
    raises(ValueError, dec.decode, table(50), dict(CFG, decoder_switch_penalty=-1.0), CTX)
    cfg_missing = {k: v for k, v in CFG.items() if k != "decoder_c_score"}
    raises(AssertionError, dec.decode, table(50), cfg_missing, CTX)

    # 10. Speed: 200k rows.
    df = table(200000, [1] * 100000 + [2] * 100000)
    df["nearest_margin_m"] = rng.normal(0, 3, len(df))
    t0 = time.perf_counter()
    dec.decode(df, CFG, CTX)
    dt = time.perf_counter() - t0
    assert dt < 10.0, dt

    # 11. Harness: decoder_v1 registered.
    assert hz.PREDICTORS["decoder_v1"] is hz.decoder_v1

    # 12. tune_decoder on a small grid, made-up IDSSE-like tables, in-process; held-out refused.
    asked = []

    def fake(mid, tables, seed):
        asked.append(mid)
        t = table(800, [1] * 400 + [2] * 400)
        away(t, 200, 400)
        away(t, 600, 800)
        t["label"] = pd.Categorical(np.where(np.arange(800) % 400 < 200, "H", "A"),
                                    categories=["H", "A", "D"])
        t["ball_dropped"] = False
        return t
    real_loader, real_par, real_grid = hz.load_table, td.run_parallel, dict(td.GRID)
    hz.load_table = fake
    td.run_parallel = lambda cfg, combos, tables, seeds, workers, ids: (
        td._init(cfg, tables, seeds, ids), [td._work((cfg, c, tables, seeds, ids)) for c in combos])[1]
    try:
        cfg0 = hz.load_config()
        small = {k: [cfg0[k]] for k in td.GRID}
        small["decoder_switch_penalty"] = sorted({cfg0["decoder_switch_penalty"], 25.0, 100.0})
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "tune_log.json")
            out = td.tune(log, 1, grid=small)
            assert len(out["results"]) == 3 and set(asked) == set(hz.ef.IDSSE_TUNE), asked
            assert all(r["spell_f1_mean"] == 1.0 for r in out["results"]), out["results"]
            assert out["chosen"] == {k: cfg0[k] for k in small}       # tie -> a priori
            raises(SystemExit, td.tune, log, 1, grid=small)          # no overwrite
            raises(ValueError, td.tune, os.path.join(tmp, "h.json"), 1,
                   ids=hz.ef.IDSSE_HOLDOUT[:1], grid=small)
            assert not set(asked) & set(hz.ef.IDSSE_HOLDOUT)
    finally:
        hz.load_table, td.run_parallel = real_loader, real_par
        td.GRID.clear(); td.GRID.update(real_grid)
    rs = [dict(a=0, spell_f1_mean=0.500), dict(a=2, spell_f1_mean=0.4995), dict(a=1, spell_f1_mean=0.499)]
    assert td.choose(rs, {"a": [0, 1, 2]}, {"a": 1})["a"] == 1
    assert td.choose(rs, {"a": [0, 1, 2]}, {"a": 2})["a"] == 2

    # 13. make_submission on the synthetic match: 199,319 rows, frames 0..N-1 in order, X exactly
    # on ungraded frames, valid per the official checker.
    env = dict(os.environ, MTAH_DATA_DIR=_dir)
    if not os.path.exists(os.path.join(_dir, "derived", "features_tactical.parquet")):
        res = subprocess.run([sys.executable, os.path.join(ROOT, "features", "extract_features.py")],
                             capture_output=True, text=True, env=env)
        assert res.returncode == 0, res.stderr
    sub = os.path.join(_dir, "derived", "submission.csv")
    if os.path.exists(sub):
        os.remove(sub)
    res = subprocess.run([sys.executable, os.path.join(ROOT, "features", "make_submission.py")],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    s = pd.read_csv(sub, dtype={"label": "object"})
    sys.path.insert(0, ROOT)
    from games import half_video_blocks
    assert list(s.columns) == ["frame", "label"] and len(s) == 199319
    assert (s["frame"].to_numpy() == np.arange(199319)).all()
    graded = np.zeros(199319, bool)
    for _, a, b in half_video_blocks("BAR-ATH"):
        graded[a:b + 1] = True
    assert ((s["label"].to_numpy() == "X") == ~graded).all()
    assert set(s["label"]) <= {"H", "A", "D", "X"}
    assert (s["label"].to_numpy()[110808:113742] == "D").all()      # known pre-kickoff dead span
    print("test_possession_decoder: all checks passed ({:.2f} s decode for 200k rows)".format(dt))


if __name__ == "__main__":
    main()
