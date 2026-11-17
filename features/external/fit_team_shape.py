"""Fit the team-shape possession model (features/team_shape.py) on the IDSSE tuning matches.

    python features/external/fit_team_shape.py --out results/tuning/team_shape/team_shape_model.json

Data: Bassek, Rein, Weber et al., Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y,
CC BY 4.0. EXTERNAL data only; BAR-ATH never goes through this script. Tables are loaded only via
harness.split_ids("tune") + harness.load_table, which refuse the held-out matches.

Training rows: alive frames (label H or A) of the clean feature tables (external_data/idsse_features/)
of the 5 tuning matches; y = 1 for H. The shape columns are checked to be identical in every
degraded table (seeds SEEDS), so a model fitted on clean tables sees the same inputs there.
Fit: L2-penalised logistic regression, no intercept, Newton (IRLS):
    minimise  -sum_i [y_i log p_i + (1 - y_i) log(1 - p_i)] + L2 / 2 * |coef|^2
on features scaled by their RMS over the training rows (no centring). L2 = L2_PENALTY, a priori.
Out of fold: leave-one-match-out (fit on the other 4, predict the 5th); the fold models are stored
so that decoder tuning (tune_decoder.py run 3) uses out-of-fold p for each tuning match. The final
model is fitted on all 5. Metrics: accuracy (p > 0.5 -> H) and mean log loss, on all alive frames
(clean tables) and on alive frames where the ball is unknown in the degraded tables
(possession_decoder.ball_unknown), pooled over seeds.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
import possession_decoder as dec  # noqa: E402
import team_shape as ts  # noqa: E402

SEEDS = (0, 1, 2)
# Unit Gaussian prior on each RMS-scaled coefficient. With ~10^5 training frames per fold it does
# not shrink measurably; it only keeps the Newton system well-conditioned. Chosen before any fit.
L2_PENALTY = 1.0
NEWTON_TOL = 1e-10
NEWTON_MAX_ITER = 100


def fit(d, y, l2):
    """Newton fit. d [N, F] raw differences (NaN allowed), y [N] in {0, 1}.
    Returns {"coef", "scales", "n_iter"}; scales = RMS of each finite column."""
    y = np.asarray(y, float)
    assert d.ndim == 2 and len(d) == len(y) and set(np.unique(y)) <= {0.0, 1.0}
    assert l2 > 0, l2
    scales = np.sqrt(np.nanmean(d ** 2, axis=0))
    assert np.isfinite(scales).all() and (scales > 0).all(), "degenerate feature scale {}".format(scales)
    X = ts.scaled(d, scales)
    w = np.zeros(X.shape[1])
    for it in range(1, NEWTON_MAX_ITER + 1):
        p = ts.sigmoid(X @ w)
        g = X.T @ (p - y) + l2 * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < NEWTON_TOL:
            return {"coef": w.tolist(), "scales": scales.tolist(), "n_iter": it}
    raise RuntimeError("Newton did not converge in {} iterations".format(NEWTON_MAX_ITER))


def metrics(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    assert len(p) == len(y) and len(p) > 0
    eps = 1e-15
    pc = np.clip(p, eps, 1 - eps)
    return {"n": int(len(y)), "accuracy": float(((p > 0.5) == (y == 1)).mean()),
            "log_loss": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
            "n_p_exactly_half": int((p == 0.5).sum())}


def model_dict(f):
    return {"features": list(ts.FEATURES), "coef": f["coef"], "scales": f["scales"]}


def load_tune(ids, seeds, loader):
    """Per match: clean (d, y) on alive rows; degraded ball-unknown alive (d, y) per seed. Asserts
    the shape columns are identical in every degraded table."""
    out = {}
    for mid in ids:
        c = loader(mid, "clean", None)
        lab = np.asarray(c["label"]).astype(str)
        alive = np.isin(lab, ["H", "A"])
        dc = ts.design(c)
        unk = []
        for s in seeds:
            g = loader(mid, "degraded", s)
            assert len(g) == len(c) and (g["frame_id"].to_numpy() == c["frame_id"].to_numpy()).all(), \
                "{} s{}: rows differ from the clean table".format(mid, s)
            for col in ts.columns():
                assert np.array_equal(g[col].to_numpy(float), c[col].to_numpy(float), equal_nan=True), \
                    "{} s{}: shape column {} differs from the clean table".format(mid, s, col)
            assert (np.asarray(g["label"]).astype(str) == lab).all(), "{} s{}: labels differ".format(mid, s)
            m = alive & dec.ball_unknown(g)
            unk.append(m)
        y = (lab == "H").astype(float)
        out[mid] = {"d": dc[alive], "y": y[alive], "d_all": dc, "y_all": y, "unknown_alive": unk,
                    "counts": {"n_rows": int(len(c)), "n_alive": int(alive.sum()),
                               "n_H": int((lab == "H").sum()), "n_A": int((lab == "A").sum()),
                               "n_ball_unknown_alive_degraded": [int(m.sum()) for m in unk]}}
    return out


def run(ids, seeds, loader, l2):
    data = load_tune(ids, seeds, loader)
    folds, per_match, p_all, y_all, p_unk, y_unk = {}, {}, [], [], [], []
    for mid in ids:
        rest = [m for m in ids if m != mid]
        f = fit(np.concatenate([data[m]["d"] for m in rest]),
                np.concatenate([data[m]["y"] for m in rest]), l2)
        folds[mid] = dict(model_dict(f), trained_on=rest, n_iter=f["n_iter"])
        p = ts.sigmoid(ts.scaled(data[mid]["d_all"], f["scales"]) @ np.asarray(f["coef"]))
        pu = np.concatenate([p[m] for m in data[mid]["unknown_alive"]])
        yu = np.concatenate([data[mid]["y_all"][m] for m in data[mid]["unknown_alive"]])
        pa = ts.sigmoid(ts.scaled(data[mid]["d"], f["scales"]) @ np.asarray(f["coef"]))
        per_match[mid] = {"alive_clean": metrics(pa, data[mid]["y"]),
                          "ball_unknown_alive_degraded": metrics(pu, yu)}
        p_all.append(pa); y_all.append(data[mid]["y"]); p_unk.append(pu); y_unk.append(yu)
    final = fit(np.concatenate([data[m]["d"] for m in ids]),
                np.concatenate([data[m]["y"] for m in ids]), l2)
    lomo = {"pooled": {"alive_clean": metrics(np.concatenate(p_all), np.concatenate(y_all)),
                       "ball_unknown_alive_degraded": metrics(np.concatenate(p_unk), np.concatenate(y_unk))},
            "per_match": per_match}
    return final, folds, lomo, {m: data[m]["counts"] for m in ids}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("{} exists: refusing to overwrite a fitted model".format(a.out))
    ids = hz.split_ids("tune", False)               # refuses held-out ids
    final, folds, lomo, counts = run(ids, SEEDS, hz.load_table, L2_PENALTY)
    data_sha = {m: hz.sha256(os.path.join(hz.EXT, "idsse_features", m + ".parquet")) for m in ids}
    data_sha.update({"{}_s{}".format(m, s): hz.sha256(os.path.join(hz.EXT, "idsse_degraded",
                                                                    "{}_s{}.parquet".format(m, s)))
                     for m in ids for s in SEEDS})
    out = dict(model_dict(final), status="done",
               timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
               l2_penalty=L2_PENALTY, n_iter=final["n_iter"], training_ids=list(ids),
               held_out_ids_not_used=list(hz.ef.IDSSE_HOLDOUT), seeds_for_metrics=list(SEEDS),
               training_rows="alive frames (label H or A) of the clean tables; y = 1 for H",
               per_match_counts=counts, lomo_metrics=lomo, lomo_folds=folds,
               script_sha256=hz.sha256(os.path.abspath(__file__)),
               team_shape_sha256=hz.sha256(ts.__file__), harness_sha256=hz.sha256(hz.__file__),
               data_sha256=data_sha, source=hz.SOURCE)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "x") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"coef": dict(zip(ts.FEATURES, final["coef"])), "lomo_pooled": lomo["pooled"],
                      "per_match_counts": counts}, indent=1))
    print("data: " + hz.SOURCE)


if __name__ == "__main__":
    sys.exit(main())
