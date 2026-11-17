"""Grid-tune the possession decoder's decoder_* keys on IDSSE (run 2026-11-16, log results/tuning/decoder_tune/tune_log.json).

    python features/external/tune_decoder.py --log results/tuning/decoder_tune/tune_log.json --workers 7
    python features/external/tune_decoder.py --run2 --log results/tuning/decoder_tune/tune_log_run2.json --workers 7
    python features/external/tune_decoder.py --run3 --log results/tuning/decoder_tune/tune_log_run3.json --workers 7

Run 2 widens only the keys run 1 chose at a grid edge (GRID_RUN2), with every other key, and the
a priori point, at the run-1 chosen values (config checked against the run-1 log first).
Run 3 tunes only decoder_shape_weight (GRID_RUN3), every other key at its run-2 frozen value
(config checked against the run-2 log first). Every run passes team-shape p_home to the decoder
from the LEAVE-ONE-MATCH-OUT fold model of each tuning match (lomo_folds in the
decoder_shape_model JSON), never the final model fitted on all five, so no in-sample leakage.

Data: Bassek, Rein, Weber et al., Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y,
CC BY 4.0. EXTERNAL data only; BAR-ATH never goes through this script.

Tuning data: the degraded (a priori ball-loss) tables of the tuning matches (ef.IDSSE_TUNE) x
SEEDS, loaded only through harness.split_ids("tune") + harness.run; held-out ids are refused.
Objective and tie rule: OBJECTIVE, TIE_RULE, declared here before the first run and written into the
log before any score. Every dead_* key keeps its config value; the detector runs once per table
(its output does not depend on decoder_* keys) and is reused across the grid.
"""

import argparse
import concurrent.futures as cf
import datetime
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
import possession_decoder as pd_  # noqa: E402
import team_shape as ts  # noqa: E402
import tune_dead_ball  # noqa: E402

SEEDS = (0, 1, 2)
GRID = {"decoder_margin_full_m": [1.0, 2.0, 4.0],
        "decoder_c_score": [0.0, 0.5, 1.0],
        "decoder_switch_penalty": [25.0, 50.0, 100.0],
        "decoder_c_entry_penalty": [12.0, 25.0, 50.0],
        "decoder_c_exit_penalty": [12.0, 25.0, 50.0],
        "decoder_restart_bias": [0.0, 50.0, 100.0]}
# Run 2 (2026-11-16): run 1 chose the low edge for these three keys. Each grid adds two halvings
# below it and keeps the run-1 neighbour above. Declared before run 2 was started.
GRID_RUN2 = {"decoder_margin_full_m": [0.25, 0.5, 1.0, 2.0],
             "decoder_c_entry_penalty": [3.0, 6.0, 12.0, 25.0],
             "decoder_c_exit_penalty": [3.0, 6.0, 12.0, 25.0]}
# Run 3 (2026-11-16): weight of the team-shape score on ball-unknown live frames, in units of a
# clear ball-known frame (+-1). A priori 0.5: shape is a weaker, team-level cue than who is nearest
# the ball, so a confident shape frame counts half a clear ball frame; with C costs ~28 a switch
# inside a ball gap then needs ~4-5 s of consistent shape evidence at p = 0.75 (about how long a
# team takes to reorganise after a turnover). Grid: 0 (= decoder v1) and factor-2 steps, two
# halvings and two doublings around 0.5. Declared before the team-shape fit and before run 3.
GRID_RUN3 = {"decoder_shape_weight": [0.0, 0.125, 0.25, 0.5, 1.0, 2.0]}
SPELL_TIE_BAND = 0.002
OBJECTIVE = ("maximise mean spell_f1_mean over the degraded runs (tuning matches x seeds {}), "
             "harness.run -> pm.evaluate").format(list(SEEDS))
TIE_RULE = ("among combos within {} of the best mean spell_f1_mean, the one closest to the a priori "
            "values (sum over keys of |grid index - a priori grid index|), then the first in grid "
            "order").format(SPELL_TIE_BAND)
METRICS = ("spell_f1_mean", "macro_f1", "team_acc_gt_alive", "n_pred_spells", "n_gt_spells")
assert tuple(GRID) == pd_.KEYS, "GRID keys must be exactly possession_decoder.KEYS"

_TABLES = {}   # per worker: (match, tables, seed) -> (table, dead_ball.detect output, LOMO p_home)
_CUR = []      # key of the table harness.run loaded last


def _init(cfg, tables, seeds, ids):
    _TABLES.clear()
    folds = ts.load(cfg)["lomo_folds"]
    for mid in hz.split_ids("tune", False, ids):
        if mid in folds[mid]["trained_on"]:
            raise ValueError("fold model for {} saw it".format(mid))
        for s in (seeds if tables == "degraded" else (None,)):
            df = hz.load_table(mid, tables, s)
            clean = df.drop(columns=[c for c in hz.LEAK_COLUMNS if c in df])
            _TABLES[(mid, tables, s)] = (df, hz.dead_ball.detect(clean, cfg, hz.IDSSE_CTX),
                                         ts.predict(clean, folds[mid]))


def _cached(mid, tables, seed):
    _CUR[:] = [(mid, tables, seed)]
    return _TABLES[(mid, tables, seed)][0]


def evaluate(cfg, combo, tables, seeds, ids):
    c = dict(cfg, **combo)

    def decoder_v1(d):
        df, det, p_home = _TABLES[_CUR[0]]
        assert len(d) == len(df) and (d.index == df.index).all(), "detector cache out of step"
        return pd_.decode_from(d, c, det, p_home)
    _, summary = hz.run(decoder_v1, "tune", tables, seeds, ids=ids, loader=_cached)
    return dict(combo, **{k: float(summary[k]["mean"]) for k in METRICS})


def _work(args):
    return evaluate(*args)


def run_parallel(cfg, combos, tables, seeds, workers, ids):
    with cf.ProcessPoolExecutor(workers, initializer=_init, initargs=(cfg, tables, seeds, ids)) as ex:
        return list(ex.map(_work, [(cfg, c, tables, seeds, ids) for c in combos], chunksize=10))


def combos_of(grid):
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


def apriori_distance(row, grid, apriori):
    return sum(abs(grid[k].index(row[k]) - grid[k].index(apriori[k])) for k in grid)


def choose(results, grid, apriori):
    """TIE_RULE. results in grid order."""
    best = max(r["spell_f1_mean"] for r in results)
    near = [r for r in results if r["spell_f1_mean"] >= best - SPELL_TIE_BAND]
    return min(near, key=lambda r: apriori_distance(r, grid, apriori))   # min is first on ties


def tune(log_path, workers, ids=None, grid=GRID, prior_log=None):
    if os.path.exists(log_path):
        raise SystemExit("{} exists: refusing to overwrite a tune log".format(log_path))
    ids = hz.split_ids("tune", False, ids)          # refuses held-out ids
    cfg = hz.frozen_config(decoder=False)
    prior = None if prior_log is None else tune_dead_ball.frozen_from_log(prior_log, cfg)
    apriori = {k: cfg[k] for k in grid}
    for k in grid:
        assert apriori[k] in grid[k], "a priori {}={} not in its grid".format(k, apriori[k])
    combos = combos_of(grid)
    head = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "grid": grid,
            "a_priori": apriori,
            "fixed": {k: cfg[k] for k in cfg if k.startswith(("dead_", "decoder_")) and k not in grid},
            "seeds": list(SEEDS), "tables": "degraded (ball_loss_variant a_priori)",
            "tune_ids": list(ids), "held_out_ids_not_used": list(hz.ef.IDSSE_HOLDOUT),
            "objective": OBJECTIVE, "tie_rule": TIE_RULE,
            "script_sha256": hz.sha256(os.path.abspath(__file__)),
            "harness_sha256": hz.sha256(hz.__file__), "decoder_sha256": hz.sha256(pd_.__file__),
            "detector_sha256": hz.sha256(hz.dead_ball.__file__),
            "team_shape_sha256": hz.sha256(ts.__file__),
            "shape_model": {"path": cfg["decoder_shape_model"],
                            "sha256": hz.sha256(os.path.join(hz.ROOT, cfg["decoder_shape_model"])),
                            "p_home": "leave-one-match-out fold models (lomo_folds)"},
            "config_sha256": hz.sha256(hz.CONFIG_PATH), "source": hz.SOURCE, "prior_log": prior}
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "x") as f:           # objective and tie rule on disk before any score
        json.dump(dict(head, status="running", n_combos=len(combos)), f, indent=1)
    results = run_parallel(cfg, combos, "degraded", SEEDS, workers, ids)
    chosen = choose(results, grid, apriori)
    log = dict(head, status="done", n_combos=len(combos), chosen={k: chosen[k] for k in grid},
               chosen_scores={m: chosen[m] for m in METRICS}, results=results)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=1)
    print(json.dumps({k: log[k] for k in ("chosen", "chosen_scores")}, indent=1))
    return log


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--run2", action="store_true", help="GRID_RUN2, after the run-1 log")
    p.add_argument("--run3", action="store_true", help="GRID_RUN3, after the run-2 log")
    a = p.parse_args()
    assert not (a.run2 and a.run3), "--run2 and --run3 are exclusive"
    if a.run3:
        tune(a.log, a.workers, grid=GRID_RUN3, prior_log=hz.DECODER_TUNE_LOG_RUN2)
    elif a.run2:
        tune(a.log, a.workers, grid=GRID_RUN2, prior_log=hz.DECODER_TUNE_LOG)
    else:
        tune(a.log, a.workers)


if __name__ == "__main__":
    sys.exit(main())
