"""Grid-tune the possession decoder's decoder_* keys on IDSSE (not yet run).

    python features/external/tune_decoder.py --log external_data/decoder_tune/tune_log.json --workers 7

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

SEEDS = (0, 1, 2)
GRID = {"decoder_margin_full_m": [1.0, 2.0, 4.0],
        "decoder_c_score": [0.0, 0.5, 1.0],
        "decoder_switch_penalty": [25.0, 50.0, 100.0],
        "decoder_c_entry_penalty": [12.0, 25.0, 50.0],
        "decoder_c_exit_penalty": [12.0, 25.0, 50.0],
        "decoder_restart_bias": [0.0, 50.0, 100.0]}
SPELL_TIE_BAND = 0.002
OBJECTIVE = ("maximise mean spell_f1_mean over the degraded runs (tuning matches x seeds {}), "
             "harness.run -> pm.evaluate").format(list(SEEDS))
TIE_RULE = ("among combos within {} of the best mean spell_f1_mean, the one closest to the a priori "
            "values (sum over keys of |grid index - a priori grid index|), then the first in grid "
            "order").format(SPELL_TIE_BAND)
METRICS = ("spell_f1_mean", "macro_f1", "team_acc_gt_alive", "n_pred_spells", "n_gt_spells")
assert tuple(GRID) == pd_.KEYS, "GRID keys must be exactly possession_decoder.KEYS"

_TABLES = {}   # per worker: (match, tables, seed) -> (table, dead_ball.detect output)
_CUR = []      # key of the table harness.run loaded last


def _init(cfg, tables, seeds, ids):
    _TABLES.clear()
    for mid in hz.split_ids("tune", False, ids):
        for s in (seeds if tables == "degraded" else (None,)):
            df = hz.load_table(mid, tables, s)
            clean = df.drop(columns=[c for c in hz.LEAK_COLUMNS if c in df])
            _TABLES[(mid, tables, s)] = (df, hz.dead_ball.detect(clean, cfg, hz.IDSSE_CTX))


def _cached(mid, tables, seed):
    _CUR[:] = [(mid, tables, seed)]
    return _TABLES[(mid, tables, seed)][0]


def evaluate(cfg, combo, tables, seeds, ids):
    c = dict(cfg, **combo)

    def decoder_v1(d):
        df, det = _TABLES[_CUR[0]]
        assert len(d) == len(df) and (d.index == df.index).all(), "detector cache out of step"
        return pd_.decode_from(d, c, det)
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


def tune(log_path, workers, ids=None, grid=GRID):
    if os.path.exists(log_path):
        raise SystemExit("{} exists: refusing to overwrite a tune log".format(log_path))
    ids = hz.split_ids("tune", False, ids)          # refuses held-out ids
    cfg = hz.frozen_config()
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
            "config_sha256": hz.sha256(hz.CONFIG_PATH), "source": hz.SOURCE}
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
    a = p.parse_args()
    tune(a.log, a.workers)


if __name__ == "__main__":
    sys.exit(main())
