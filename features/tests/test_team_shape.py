"""Synthetic checks of features/team_shape.py, features/external/fit_team_shape.py and the
team-shape hook in features/possession_decoder.py. Made-up tables only.

    python features/tests/test_team_shape.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external"))
import fit_team_shape as fts  # noqa: E402
import possession_decoder as dec  # noqa: E402
import team_shape as ts  # noqa: E402

MODEL_TOY = {"features": list(ts.FEATURES), "coef": [1.0, -0.5, 0.3, 0.2, -0.1, 0.4, 0.7, -0.6],
             "scales": [10.0, 1.0, 0.3, 5.0, 5.0, 8.0, 1.0, 0.3]}
CFG = {"decoder_margin_full_m": 2.0, "decoder_c_score": 1.0, "decoder_switch_penalty": 50.0,
       "decoder_c_entry_penalty": 25.0, "decoder_c_exit_penalty": 25.0, "decoder_restart_bias": 50.0,
       "decoder_shape_weight": 0.0, "decoder_shape_model": "unused-by-decode_from"}


def shape_table(rng, n, d=None):
    """Random h_/a_ shape columns; if d [n, F] is given, h - a equals it exactly (a = 0)."""
    cols = {}
    for j, f in enumerate(ts.FEATURES):
        if d is None:
            cols["h_" + f], cols["a_" + f] = rng.normal(0, 3, n), rng.normal(0, 3, n)
        else:
            cols["h_" + f], cols["a_" + f] = d[:, j], np.zeros(n)
    return pd.DataFrame(cols)


def swapped(df):
    out = df.copy()
    for f in ts.FEATURES:
        out["h_" + f], out["a_" + f] = df["a_" + f], df["h_" + f]
    return out


def live(n, rng):
    """Decoder table: home clearly closer, ball known, live play, plus shape columns."""
    df = pd.DataFrame({"half": np.ones(n, np.int8), "ball_x_m": np.full(n, 50.0),
                       "nearest_team": np.zeros(n, np.int8), "nearest_margin_m": np.full(n, 3.0),
                       "h_n_within_3m": np.ones(n), "a_n_within_3m": np.zeros(n)})
    return pd.concat([df, shape_table(rng, n)], axis=1)


def det(n):
    return {"dead": np.zeros(n, bool), "segments": [], "restart_team": []}


def runs(lab):
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    e = np.concatenate(([0], b, [len(lab)]))
    return [(int(s), int(t), lab[s]) for s, t in zip(e[:-1], e[1:])]


def main():
    rng = np.random.default_rng(0)

    # 1. No ball-derived column among the model's inputs; inputs are exactly h_/a_ of FEATURES.
    for c in ts.columns():
        assert not any(tok in c for tok in ts.BALL_TOKENS), c
    for bad in ("ball_x_m", "nearest_margin_m", "h_dmin_ball_m", "a_d2_ball_m", "h_n_within_3m"):
        assert bad not in ts.columns()
    assert sorted(ts.columns()) == sorted(p + f for f in ts.FEATURES for p in ("h_", "a_"))

    # 2. Antisymmetry: swapping the teams gives exactly 1 - p; all-NaN row -> 0.5; NaN feature
    # contributes 0.
    df = shape_table(rng, 1000)
    p, q = ts.predict(df, MODEL_TOY), ts.predict(swapped(df), MODEL_TOY)
    assert np.abs(p + q - 1).max() < 1e-15, np.abs(p + q - 1).max()
    df.loc[0, ts.columns()] = np.nan
    df.loc[1, "h_adv_x_m"] = np.nan
    p = ts.predict(df, MODEL_TOY)
    assert p[0] == 0.5
    z = sum(MODEL_TOY["coef"][j] * (df.loc[1, "h_" + f] - df.loc[1, "a_" + f]) / MODEL_TOY["scales"][j]
            for j, f in enumerate(ts.FEATURES) if f != "adv_x_m")
    assert abs(p[1] - 1 / (1 + np.exp(-z))) < 1e-12
    try:
        ts.predict(df, dict(MODEL_TOY, features=list(ts.FEATURES[::-1])))
        raise AssertionError("feature-order mismatch not refused")
    except AssertionError as e:
        assert "model features" in str(e)

    # 3. Fit recovers known coefficients on RMS-scaled features (zero-mean, so RMS = sd).
    n, sd = 200000, np.array(MODEL_TOY["scales"])
    w = np.array(MODEL_TOY["coef"])
    d = rng.normal(0, 1, (n, len(w))) * sd
    y = (rng.random(n) < ts.sigmoid((d / sd) @ w)).astype(float)
    d_nan = d.copy()
    d_nan[rng.random(d.shape) < 0.02] = np.nan               # NaN tolerated (contributes 0)
    f = fts.fit(d, y, fts.L2_PENALTY)
    assert np.abs(np.array(f["coef"]) - w).max() < 0.05, f["coef"]
    assert np.abs(np.array(f["scales"]) / sd - 1).max() < 0.01, f["scales"]
    f2 = fts.fit(d_nan, y, fts.L2_PENALTY)
    assert np.isfinite(f2["coef"]).all()
    m = fts.metrics(ts.sigmoid((d / sd) @ w), y)
    assert 0.5 < m["accuracy"] < 1 and m["log_loss"] < np.log(2)

    # 4. run() with made-up matches: LOMO folds exclude their own match, counts add up, and a
    # degraded table whose shape columns differ is refused.
    ids = ("M1", "M2", "M3")
    tables = {}
    for k, mid in enumerate(ids):
        dd = rng.normal(0, 1, (3000, len(w))) * sd
        t = shape_table(rng, 3000, dd)
        yy = rng.random(3000) < ts.sigmoid((dd / sd) @ w)
        t["label"] = np.where(np.arange(3000) % 10 == 0, "D", np.where(yy, "H", "A"))
        t["frame_id"] = np.arange(3000)
        t["ball_x_m"] = np.where(np.arange(3000) % 3 == 0, np.nan, 1.0)
        t["nearest_margin_m"] = 1.0
        tables[mid] = t

    def loader(mid, kind, seed):
        return tables[mid]
    final, folds, lomo, counts = fts.run(ids, (0, 1), loader, fts.L2_PENALTY)
    for mid in ids:
        assert mid not in folds[mid]["trained_on"] and len(folds[mid]["trained_on"]) == 2
        c = counts[mid]
        assert c["n_alive"] == c["n_H"] + c["n_A"] == 2700
        assert c["n_ball_unknown_alive_degraded"] == [900, 900]
    assert lomo["pooled"]["alive_clean"]["n"] == 8100
    assert lomo["pooled"]["ball_unknown_alive_degraded"]["n"] == 2 * 2700
    assert lomo["pooled"]["alive_clean"]["accuracy"] > 0.6

    def bad_loader(mid, kind, seed):
        if kind == "clean":
            return tables[mid]
        g = tables[mid].copy()
        g.loc[5, "h_spread_m"] += 1.0
        return g
    try:
        fts.run(ids, (0,), bad_loader, fts.L2_PENALTY)
    except ValueError as e:
        assert "shape column" in str(e), e
    else:
        raise AssertionError("differing degraded shape column not refused")

    # 5. Decoder hook: weight 0 reproduces decoder v1 (ball-unknown team score 0) for any p_home.
    n = 3000
    df = live(n, rng)
    df["nearest_margin_m"] = rng.normal(0, 3, n)
    gap = (np.arange(n) // 300) % 2 == 1                       # ball unknown in alternate 12 s blocks
    df.loc[gap, ["ball_x_m", "nearest_margin_m", "h_n_within_3m", "a_n_within_3m"]] = np.nan
    df.loc[gap, "nearest_team"] = -1
    p_rand = rng.random(n)
    e = dec.emissions(df, CFG, p_rand)
    assert (e[gap] == 0).all()
    m_ = df["nearest_margin_m"].to_numpy()
    assert (e[~gap, 1] == np.clip(m_[~gap] / 2.0, -1, 1)).all()
    lab0 = dec.decode_from(df, CFG, det(n), np.full(n, 0.5))
    assert (dec.decode_from(df, CFG, det(n), p_rand) == lab0).all()
    # ... and ball-known frames never change with the weight.
    e2 = dec.emissions(df, dict(CFG, decoder_shape_weight=2.0), p_rand)
    assert (e2[~gap] == e[~gap]).all() and np.allclose(e2[gap, 1], 2.0 * (2 * p_rand[gap] - 1))
    assert (e2[gap, 0] == 0).all() and (e2[:, 2] == -e2[:, 1]).all()

    # 6. Decoder hook: home spell, ball lost 200..800, shape says away -> weight 1 switches to A in
    # the gap, weight 0 carries H.
    n = 1000
    df = live(n, rng)
    df.loc[200:799, ["ball_x_m", "nearest_margin_m", "h_n_within_3m", "a_n_within_3m"]] = np.nan
    df.loc[200:799, "nearest_team"] = -1
    p = np.full(n, 0.5)
    p[200:800] = 0.05
    assert runs(dec.decode_from(df, CFG, det(n), p)) == [(0, 1000, "H")]
    r = runs(dec.decode_from(df, dict(CFG, decoder_shape_weight=1.0), det(n), p))
    assert [x[2] for x in r] == ["H", "A", "H"] and abs(r[1][0] - 200) <= 30 and abs(r[1][1] - 800) <= 30, r

    # 7. Guards: negative weight, p_home out of range or wrong length.
    for bad_cfg, bad_p in ((dict(CFG, decoder_shape_weight=-1.0), p), (CFG, p + 1.0), (CFG, p[:-1])):
        try:
            dec.emissions(df, bad_cfg, bad_p)
            raise RuntimeError("not refused")
        except AssertionError:
            pass
    print("test_team_shape: all checks passed")


if __name__ == "__main__":
    main()
