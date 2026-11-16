"""IDSSE tuning harness: score a per-frame H/A/D predictor on IDSSE feature tables.

    python features/external/harness.py --predictor baseline_nearest --tables degraded --seeds 0 1 2
    python features/external/harness.py --predictor baseline_nearest --tables clean
    python features/external/harness.py --predictor baseline_nearest --holdout ...   # evaluation only

Data: Bassek, Rein, Weber et al., Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y,
CC BY 4.0. EXTERNAL data only; the provided BAR-ATH match never goes through this script.

Scoring is evaluation/possession_metrics.py (pm.label_codes, pm.evaluate), nothing else. `label`
is the DFL proxy (idsse_features.py). The two periods are adjacent rows, so SEAM_ROWS X rows are
inserted at the seam in both GT and prediction: pm.valid_runs then splits there, no spell crosses
halftime, and the gap exceeds the widest transition tolerance (pm.match_transitions assumes runs
far apart); no real frame is dropped.

Split (CLAUDE.md, fixed 2026-11-09): "tune" = ef.IDSSE_TUNE. "holdout" = ef.IDSSE_HOLDOUT, for
evaluation only: it needs allow_holdout=True (CLI --holdout) and every look is appended to
external_data/harness/holdout_log.json.

Predictors get the table with LEAK_COLUMNS dropped: `label` (the answer) and `ball_dropped` (dropped
for hygiene; it equals ~ball_seen, which is the intended observable).

Dead-ball edges in degraded tables: the ball-loss onset and reacquisition lags are drawn per dead
span from a priori uniform spreads (0-4 s and 0-1 s; degrade_ball.py), so the ball no longer
vanishes/reappears at a fixed offset from the DFL edges. A detector's ball vanish/reappear offsets
stay a_priori and are never tuned on degraded tables. Residual risk: the spreads are themselves
assumptions, so degraded alive/dead scores are only as realistic as they are.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "features"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import extract_features as ef  # noqa: E402
import dead_ball  # noqa: E402
import possession_metrics as pm  # noqa: E402

EXT = os.path.join(ROOT, "external_data")
CONFIG_PATH = os.path.join(ROOT, "features", "config.json")
HOLDOUT_LOG = os.path.join(EXT, "harness", "holdout_log.json")
DEAD_BALL_TUNE_LOG = os.path.join(EXT, "dead_ball_tune", "tune_log.json")
DEAD_BALL_MIN_DEAD_LOG = os.path.join(EXT, "dead_ball_tune", "tune_log_min_dead.json")
LEAK_COLUMNS = ["label", "ball_dropped"]
SEAM_ROWS = max(pm.TAUS) + 1
SHOW = ["spell_f1_mean", "spell_f1_at_headline", "macro_f1", "alive_dead_f1",
        "team_acc_gt_alive", "n_gt_spells", "n_pred_spells", "fragmentation"]
BANNER = "HELD OUT — evaluation only, do not tune or select on this"
SOURCE = "Bassek, Rein, Weber et al., Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y, CC BY 4.0"


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def with_seam(codes, half):
    """codes with SEAM_ROWS X inserted between period 1 and period 2 (periods must be contiguous)."""
    half = np.asarray(half)
    assert len(codes) == len(half), "codes {} vs half {}".format(len(codes), len(half))
    assert set(np.unique(half)) == {1, 2}, "need both periods, got {}".format(np.unique(half))
    seam = int(np.flatnonzero(half == 2)[0])
    assert (half[:seam] == 1).all() and (half[seam:] == 2).all(), "periods not contiguous 1 then 2"
    return np.insert(codes, seam, [pm.X] * SEAM_ROWS)


def codes(labels):
    """pm.label_codes on the distinct values only (it is elementwise), mapped back: same result,
    without a Python loop over every frame."""
    uniq, inv = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return pm.label_codes(uniq)[inv]


def score_match(df, pred_labels):
    """pm.evaluate on one match, GT = df['label'], with the period seam X inserted in both."""
    assert len(pred_labels) == len(df), "pred {} rows vs table {}".format(len(pred_labels), len(df))
    gt = codes(np.asarray(df["label"]).astype(str))
    assert (gt != pm.X).all(), "IDSSE GT should have no X before the seam"
    pred = codes(np.asarray(pred_labels))
    return pm.evaluate(with_seam(gt, df["half"]), with_seam(pred, df["half"]))


def baseline_nearest(df):
    """Ball position known (seen or interpolated) -> team of the nearest player (0 H, 1 A), else D."""
    known = df["ball_x_m"].notna().to_numpy()
    nt = df["nearest_team"].to_numpy()
    return np.where(known & (nt == 0), "H", np.where(known & (nt == 1), "A", "D"))


# IDSSE context for the dead-ball detector: no published dead spans (periods start at kickoff),
# no Smart Tagging.
IDSSE_CTX = {"known_dead": [], "smart_tagging": False}


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    ef.check_provenance(cfg)
    return cfg


def frozen_config():
    """load_config(), checked against both dead-ball tune logs (tune_dead_ball.frozen_from_log).
    Every path that runs the tuned detector loads its config through here."""
    import tune_dead_ball
    cfg = load_config()
    tune_dead_ball.frozen_from_log(DEAD_BALL_TUNE_LOG, cfg)
    tune_dead_ball.frozen_from_log(DEAD_BALL_MIN_DEAD_LOG, cfg)
    return cfg


def dead_ball_v1(df):
    """features/dead_ball.py v1 with the values in config.json (frozen_config)."""
    cfg = frozen_config()
    return dead_ball.predict_labels(df, cfg, IDSSE_CTX)


def decoder_v1(df):
    """features/possession_decoder.py v1 with the values in config.json (frozen_config). The
    decoder_* values are a priori until tune_decoder.py has been run."""
    import possession_decoder
    cfg = frozen_config()
    return possession_decoder.decode(df, cfg, IDSSE_CTX)


PREDICTORS = {"baseline_nearest": baseline_nearest, "dead_ball_v1": dead_ball_v1,
              "decoder_v1": decoder_v1}


def split_ids(split, allow_holdout, ids=None):
    """Match ids for a split; ids (optional subset) must lie inside it."""
    assert split in ("tune", "holdout"), "split must be tune or holdout, got {!r}".format(split)
    pool = ef.IDSSE_TUNE if split == "tune" else ef.IDSSE_HOLDOUT
    if split == "holdout" and not allow_holdout:
        raise PermissionError("held-out split needs allow_holdout=True (CLI --holdout)")
    ids = tuple(pool if ids is None else ids)
    if split == "tune" and set(ids) & set(ef.IDSSE_HOLDOUT):
        raise ValueError("tune split asked for held-out ids {}".format(
            sorted(set(ids) & set(ef.IDSSE_HOLDOUT))))
    if not set(ids) <= set(pool):
        raise ValueError("ids {} outside the {} split {}".format(ids, split, pool))
    return ids


def load_table(mid, tables, seed, allow_holdout=False):
    if mid in ef.IDSSE_HOLDOUT and not allow_holdout:
        raise PermissionError("{} is held out; only run(..., allow_holdout=True) may load it".format(mid))
    if tables == "clean":
        return pd.read_parquet(os.path.join(EXT, "idsse_features", mid + ".parquet"))
    assert tables == "degraded", "tables must be degraded or clean, got {!r}".format(tables)
    stem = os.path.join(EXT, "idsse_degraded", "{}_s{}".format(mid, seed))
    with open(stem + "_provenance.json") as f:
        variant = json.load(f)["ball_loss_variant"]
    if variant != "a_priori":
        raise ValueError("{}: ball-loss variant {!r}, expected a_priori".format(stem, variant))
    return pd.read_parquet(stem + ".parquet")


def run(predictor, split, tables, seeds, allow_holdout=False, ids=None, loader=load_table):
    """Score predictor on every (match, seed) of the split. Clean tables: one run per match.
    Returns (rows, summary): rows are dicts match/seed + SHOW metrics, summary is mean/min/max."""
    ids = split_ids(split, allow_holdout, ids)
    if split == "holdout":
        print("=" * 72 + "\n" + BANNER + "\n" + "=" * 72)
    rows = []
    for mid in ids:
        for seed in (seeds if tables == "degraded" else (None,)):
            df = (loader(mid, tables, seed, allow_holdout=True) if loader is load_table
                  and split == "holdout" else loader(mid, tables, seed))
            pred = np.asarray(predictor(df.drop(columns=[c for c in LEAK_COLUMNS if c in df])))
            res = score_match(df, pred)
            rows.append(dict(match=mid, seed=seed, **{k: res[k] for k in SHOW}))
    t = pd.DataFrame(rows)[SHOW].astype(float)
    summary = {k: {"mean": t[k].mean(), "min": t[k].min(), "max": t[k].max()} for k in SHOW}
    if split == "holdout":
        log_holdout(predictor, tables, seeds, ids, summary)
    return rows, summary


def log_holdout(predictor, tables, seeds, ids, summary):
    os.makedirs(os.path.dirname(HOLDOUT_LOG), exist_ok=True)
    log = []
    if os.path.exists(HOLDOUT_LOG):
        with open(HOLDOUT_LOG) as f:
            log = json.load(f)
    log.append({"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "predictor": predictor.__name__, "script_sha256": sha256(os.path.abspath(__file__)),
                "config_sha256": sha256(CONFIG_PATH), "tables": tables,
                "seeds": list(seeds) if tables == "degraded" else None, "matches": list(ids),
                "summary": summary, "source": SOURCE})
    with open(HOLDOUT_LOG, "w") as f:
        json.dump(log, f, indent=2)


def print_table(rows, summary):
    fmt = lambda k, v: "{:>8d}".format(int(v)) if k.startswith("n_") else "{:>8.3f}".format(v)
    head = ["sf1_mean", "sf1@0.5", "macroF1", "aliveF1", "team_acc", "n_gt_sp", "n_pr_sp", "frag"]
    print("{:<8} {:>4} ".format("match", "seed") + " ".join("{:>8}".format(h) for h in head))
    for r in rows:
        print("{:<8} {:>4} ".format(r["match"], "-" if r["seed"] is None else r["seed"])
              + " ".join(fmt(k, r[k]) for k in SHOW))
    print("{:<13} ".format("mean") + " ".join("{:>8.3f}".format(summary[k]["mean"]) for k in SHOW))
    print("{:<13} ".format("min-max") + " ".join(
        "{:>8}".format("{:.2f}-{:.2f}".format(summary[k]["min"], summary[k]["max"])
                       if not k.startswith("n_") else "{:.0f}-{:.0f}".format(
                           summary[k]["min"], summary[k]["max"])) for k in SHOW))
    print("data: " + SOURCE)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictor", required=True, choices=sorted(PREDICTORS))
    p.add_argument("--holdout", action="store_true", help="held-out split, evaluation only")
    p.add_argument("--tables", required=True, choices=["degraded", "clean"])
    p.add_argument("--seeds", type=int, nargs="+", help="degraded tables only")
    a = p.parse_args()
    if a.tables == "degraded" and not a.seeds:
        p.error("--seeds required with --tables degraded")
    rows, summary = run(PREDICTORS[a.predictor], "holdout" if a.holdout else "tune", a.tables,
                        tuple(a.seeds or ()), allow_holdout=a.holdout)
    print("predictor {}  split {}  tables {}".format(
        a.predictor, "holdout" if a.holdout else "tune", a.tables))
    print_table(rows, summary)


if __name__ == "__main__":
    sys.exit(main())
