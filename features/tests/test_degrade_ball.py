"""Synthetic check of features/external/degrade_ball.py (ball-loss simulation on IDSSE; data:
Bassek, Rein, Weber et al., Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y, CC BY 4.0).
Made-up dead spans only, no real data.

    .venv/bin/python features/tests/test_degrade_ball.py
"""

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external"))
sys.path.insert(0, HERE)
import degrade_ball as db  # noqa: E402
import idsse_features as idf  # noqa: E402
import test_idsse_features as tif  # noqa: E402

FPS = 25


def params(p_drop, onset, reacq, live_rate, flick_rate, p_short=None):
    """p_short: drop prob of gaps < LONG_MIN_FRAMES; same as p_drop unless given."""
    return {"ball_loss_dead_drop_prob": p_drop, "ball_loss_onset_lag_frames": onset,
            "ball_loss_short_gap_drop_prob": p_drop if p_short is None else p_short,
            "ball_loss_reacquisition_lag_frames": reacq, "ball_loss_live_gaps_per_hour": live_rate,
            "ball_loss_live_gap_frames_range": [20, 40], "ball_loss_flickers_per_hour": flick_rate,
            "ball_loss_flicker_frames_range": [1, 3]}


def main():
    # two periods of 200 rows; dead spans [0,10) (period-1 start), [50,60), [90,91) (shorter than
    # the onset lag: skipped), [190,200) (period end: reacq lag clipped), [200,210), [300,330)
    n = 400
    period = np.repeat([1, 2], 200).astype(np.int8)
    alive = np.ones(n, dtype=np.int8)
    for a, b in ((0, 10), (50, 60), (90, 91), (190, 200), (200, 210), (300, 330)):
        alive[a:b] = 0
    exact = np.zeros(n, dtype=bool)
    for a, b in ((4, 12), (54, 62), (194, 200), (204, 212), (304, 332)):  # onset 4, reacq 2
        exact[a:b] = True
    got = db.drop_mask(alive, period, params(1.0, 4, 2, 0.0, 0.0), FPS, np.random.default_rng(7))
    assert (got == exact).all(), np.flatnonzero(got != exact)
    none = db.drop_mask(alive, period, params(0.0, 4, 2, 0.0, 0.0), FPS, np.random.default_rng(7))
    assert not none.any()
    # short-gap rule: only a gap of >= LONG_MIN_FRAMES (here [10+4, 150+0) = 136 frames) survives
    al = np.ones(400, dtype=np.int8)
    al[10:150] = 0   # gap 136 >= 126: dropped
    al[200:300] = 0  # gap 96 < 126: kept seen with p_short = 0
    want = np.zeros(400, dtype=bool)
    want[14:150] = True
    got = db.drop_mask(al, np.ones(400, np.int8), params(1.0, 4, 0, 0.0, 0.0, p_short=0.0), FPS,
                       np.random.default_rng(7))
    assert (got == want).all(), np.flatnonzero(got != want)
    assert db.LONG_MIN_FRAMES == 126
    for bad in (params(1.5, 0, 0, 0.0, 0.0), params(1.0, -1, 0, 0.0, 0.0),
                params(1.0, 0, 0, 0.0, 0.0, p_short=-0.1)):
        try:
            db.drop_mask(al, np.ones(400, np.int8), bad, FPS, np.random.default_rng(0))
        except ValueError:
            pass
        else:
            raise AssertionError("bad parameter accepted: {}".format(bad))
    for bad_lag in ([5, 2], [-1, 3], [0, 1, 2], 2.5, [0.0, 3], True):
        try:
            db.check_params(dict(params(1.0, bad_lag, 0, 0.0, 0.0)))
        except ValueError:
            pass
        else:
            raise AssertionError("bad lag accepted: {!r}".format(bad_lag))
    db.check_params(params(1.0, [0, 0], [3, 3], 0.0, 0.0))

    # lag ranges: drawn per span; offsets within [lo, hi]; more than one distinct value; each gap
    # stays inside its period and [dead start, restart + reacq); seeded
    n_sp, span, step = 200, 300, 600  # 200 dead spans of 300 frames, 300 alive frames between
    alive = np.ones(n_sp * step, dtype=np.int8)
    for i in range(n_sp):
        alive[i * step:i * step + span] = 0
    period = np.repeat([1, 2], n_sp * step // 2).astype(np.int8)
    pr = params(1.0, [0, 100], [0, 25], 0.0, 0.0)
    m = db.drop_mask(alive, period, pr, FPS, np.random.default_rng(11))
    st, sp = db.true_runs(m)
    assert len(st) == n_sp, len(st)  # one gap per dead span, none merged
    on, re = st - np.arange(n_sp) * step, sp - (np.arange(n_sp) * step + span)
    assert on.min() >= 0 and on.max() <= 100 and re.min() >= 0 and re.max() <= 25, (on, re)
    assert len(set(on)) > 1 and len(set(re)) > 1
    assert (m == db.drop_mask(alive, period, pr, FPS, np.random.default_rng(11))).all()
    assert (m != db.drop_mask(alive, period, pr, FPS, np.random.default_rng(12))).any()
    # reacquisition never runs past the period end; gap classified after the draw: a 130-frame
    # dead span at the end of period 1 with onset in [0, 10] stays long (>= 126) and is dropped
    # with p_short = 0 only when the drawn onset keeps it >= 126 frames
    al = np.ones(400, dtype=np.int8)
    al[70:200] = 0
    al[300:330] = 0  # 30-frame span: short whatever the lags, p_short = 0 -> never dropped
    per = np.repeat([1, 2], 200).astype(np.int8)
    for seed in range(40):
        rng_lags = np.random.default_rng(seed)
        g = db.drop_mask(al, per, params(1.0, [0, 10], [0, 25], 0.0, 0.0, p_short=0.0), FPS,
                         np.random.default_rng(seed))
        onset = rng_lags.integers(0, 11, 1)[0]  # first draw of period 1: its one span's onset
        assert not g[200:].any() and not g[:70].any()
        assert g[70 + onset:200].all() == (200 - 70 - onset >= db.LONG_MIN_FRAMES)
        assert g[70:200].sum() in (0, 200 - 70 - onset)

    # still_after_reappear: within the period only, NaN-aware
    msk = np.zeros(400, dtype=bool)
    msk[10:150] = True    # ends at 150, row 162 checked
    msk[200:390] = True   # ends at 390, 390 + 12 > 400: outside the period, skipped
    sf = np.zeros(400)
    sf[162] = 1.0
    assert db.still_after_reappear(msk, np.ones(400, np.int8), sf) == 1.0
    sf[162] = np.nan
    assert np.isnan(db.still_after_reappear(msk, np.ones(400, np.int8), sf))
    # per-match streams differ for the same seed
    assert db.match_rng(0, "J03WMX").random() != db.match_rng(0, "J03WN1").random()

    # live gaps and flickers on a long all-alive match: lengths in range, never touching each
    # other, same seed -> same mask, other seed -> different mask
    n = 2 * 90000
    period = np.repeat([1, 2], n // 2).astype(np.int8)
    alive = np.ones(n, dtype=np.int8)
    alive[1000:1300] = 0
    p = params(1.0, 0, 0, 30.0, 20.0)  # 2 h -> about 60 live gaps and 40 flickers
    m = db.drop_mask(alive, period, p, FPS, np.random.default_rng(3))
    assert (m == db.drop_mask(alive, period, p, FPS, np.random.default_rng(3))).all()
    assert (m != db.drop_mask(alive, period, p, FPS, np.random.default_rng(4))).any()
    L = db.missing_runs(m, period)
    assert ((L <= 3) | ((L >= 20) & (L <= 40)) | (L == 300)).all(), np.unique(L)
    n_live, n_flick = ((L >= 20) & (L <= 40)).sum(), (L <= 3).sum()
    assert 35 < n_live < 85 and 20 < n_flick < 60, (n_live, n_flick)
    assert (L == 300).sum() == 1, "dead span must stay one run (placement keeps a free margin)"
    assert m[1000:1300].all() and not m[999] and not m[1300]
    agg = db.aggregates(m, period, FPS)
    assert agg["long_runs"] == 1 and agg["short_runs_1_12"] == n_flick

    # no room -> fail loudly
    try:
        db.drop_mask(np.ones(100, np.int8), np.ones(100, np.int8), params(1.0, 0, 0, 1e6, 0.0),
                     FPS, np.random.default_rng(0))
    except db.NoRoom:
        pass
    else:
        raise AssertionError("placing more gaps than fit must fail")

    # tune() loads only the five tuning matches and writes its log
    seen_ids = []

    def fake_load(mid, in_dir):
        seen_ids.append(mid)
        a = np.ones(2000, dtype=np.int8)
        a[100:400] = 0
        return a, np.repeat([1, 2], 1000).astype(np.int8), None, None
    real_load = db.load
    db.load = fake_load
    try:
        with open(os.path.join(HERE, "..", "config.json")) as f:
            cfg0 = dict(json.load(f), ball_loss_live_gaps_per_hour=0.0)  # fits the tiny match
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "tune_log.json")
            grid = {"ball_loss_dead_drop_prob": [0.5, 1.0]}
            rows = db.tune("unused", cfg0, [0], grid, log, 1)
            with open(log) as f:
                logged = json.load(f)
    finally:
        db.load = real_load
    assert seen_ids == list(db.ef.IDSSE_TUNE), seen_ids
    assert not set(seen_ids) & set(db.ef.IDSSE_HOLDOUT)
    assert len(rows) == 2 and len(logged["results"]) == 2 and logged["timestamp"]
    assert logged["tuned_on"] == list(db.ef.IDSSE_TUNE) and logged["infeasible"] == 0

    # ball_override goes through the shared extractor: ball_seen == ~mask, labels unchanged
    with open(os.path.join(HERE, "..", "config.json")) as f:
        cfg = json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        expected, (_, _, ball, period) = tif.make_match(tmp, np.random.default_rng(0))
        npz, js = os.path.join(tmp, "SYN.npz"), os.path.join(tmp, "SYN.json")
        mask = np.zeros(len(ball), dtype=bool)
        mask[100:150] = True
        mask[200:205] = True
        b = ball.astype(np.float32).copy()
        b[mask] = np.nan
        df, _ = idf.match_features(npz, js, cfg, "after", ball_override=b)
        assert (df["ball_seen"].values == ~mask).all()
        assert (df["label"].astype(str).values == expected).all()
        clean, _ = idf.match_features(npz, js, cfg, "after")
        assert clean["ball_seen"].all() and list(clean.columns) == list(df.columns)
        try:
            idf.match_features(npz, js, cfg, "after", ball_override=b[:-1])
        except AssertionError as e:
            assert "shape" in str(e), e
        else:
            raise AssertionError("wrong-shape ball_override must fail")

    # --ball-loss-params: the fitted variant replaces exactly the 8 keys and their provenance and
    # still passes check_provenance; an a priori build needs no tune log, a tuned one does
    fitted_path = os.path.join(HERE, "..", "external", "ball_loss_fitted_bar_ath.json")
    with open(fitted_path) as f:
        fitted = json.load(f)
    fc = db.with_ball_loss_params(cfg, fitted_path)
    db.ef.check_provenance(fc)
    for k in db.PARAM_KEYS:
        assert fc[k] == fitted[k] and fc["_provenance"][k] == fitted["_provenance"][k], k
    assert {k: v for k, v in fc.items() if k not in db.PARAM_KEYS and k != "_provenance"} == \
        {k: v for k, v in cfg.items() if k not in db.PARAM_KEYS and k != "_provenance"}
    assert all(cfg["_provenance"][k]["basis"] != "tuned_external" for k in db.PARAM_KEYS)
    with tempfile.TemporaryDirectory() as tmp:
        absent_log = os.path.join(tmp, "tune_log.json")
        assert db.variant_stamp(cfg, None, absent_log) == {"ball_loss_variant": "a_priori"}
        try:
            db.variant_stamp(fc, fitted_path, absent_log)
        except SystemExit:
            pass
        else:
            raise AssertionError("a tuned_external build without tune_log.json must fail")
        bad = dict(fitted, ball_loss_extra=1)
        with open(os.path.join(tmp, "bad.json"), "w") as f:
            json.dump(bad, f)
        try:
            db.with_ball_loss_params(cfg, os.path.join(tmp, "bad.json"))
        except ValueError:
            pass
        else:
            raise AssertionError("a params file with an extra key must fail")
    print("test_degrade_ball: all checks passed")


if __name__ == "__main__":
    main()
