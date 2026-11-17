"""Grid-tune the dead-ball detector's player-based keys (dead_ball.TUNABLE) on IDSSE.

    python features/external/tune_dead_ball.py --tune --log external_data/dead_ball_tune/tune_log.json --workers 7
    python features/external/tune_dead_ball.py --crosscheck --log ... --out ... --workers 7
    python features/external/tune_dead_ball.py --check --log ...      # config == log's chosen row
    python features/external/tune_dead_ball.py --tune-one dead_min_dead_frames \
        --log external_data/dead_ball_tune/tune_log_min_dead.json --workers 5

Data: Bassek, Rein, Weber et al., Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y,
CC BY 4.0. EXTERNAL data only; BAR-ATH never goes through this script.

Tuning data: the degraded (a priori ball-loss) tables of the 5 tuning matches (ef.IDSSE_TUNE)
x SEEDS, loaded only through harness.run(split="tune"). Held-out tables are never loaded. Objective
and tie rule: OBJECTIVE, TIE_RULE (declared before the first run, written into the log). Scoring is
harness.run -> pm.evaluate; nothing is re-implemented here. Every non-TUNABLE dead_* key keeps its
config value (the ball vanish/reappear offsets are never tuned).
"""

import argparse
import concurrent.futures as cf
import datetime
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402

SEEDS = (0, 1, 2)
GRID = {"dead_decel_lag_frames": [0, 12, 25, 50, 75],
        "dead_smooth_half_window_frames": [6, 12, 18, 25],
        "dead_walking_frac_min": [0.6, 0.7, 0.8, 0.9, 0.95],
        "dead_mean_speed_max_m_s": [1.0, 1.25, 1.5, 2.0, 2.5],
        "dead_min_gap_frames": [25, 50, 100, 150],
        "dead_min_alive_frames": [0, 12, 25, 50, 75]}
SPELL_TIE_BAND = 0.002
OBJECTIVE = ("maximise mean spell_f1_mean over the {} degraded runs (5 tuning matches x seeds {}), "
             "harness.run -> pm.evaluate").format(5 * len(SEEDS), list(SEEDS))
TIE_RULE = ("among combos within {} of the best mean spell_f1_mean, pick the highest mean "
            "alive_dead_f1; if still tied (equal to 1e-12), the one closest to the a priori values "
            "(sum over keys of |grid index - a priori grid index|), then the first in grid order"
            ).format(SPELL_TIE_BAND)
METRICS = ("spell_f1_mean", "alive_dead_f1", "macro_f1")
# One-key searches (--tune-one), run after the joint grid with every other dead_* key held at its
# config value. Grid and tie rule declared 2026-11-16 (user decision) before the first score.
ONE_KEY_GRID = {"dead_min_dead_frames": [0, 12, 25, 38, 50]}
ONE_KEY_TIE_BAND = 0.001
ONE_KEY_TIE_RULE = ("highest mean spell_f1_mean; if the top values are within {} of each other, "
                    "take the smallest value (frame count) among them").format(ONE_KEY_TIE_BAND)
assert tuple(GRID) + tuple(ONE_KEY_GRID) == hz.dead_ball.TUNABLE, \
    "GRID + ONE_KEY_GRID keys must be exactly dead_ball.TUNABLE"

_TABLES = {}   # per worker process: (match, tables, seed) -> DataFrame


def _init(tables, seeds):
    for mid in hz.split_ids("tune", False):
        for s in (seeds if tables == "degraded" else (None,)):
            _TABLES[(mid, tables, s)] = hz.load_table(mid, tables, s)


def _cached(mid, tables, seed):
    return _TABLES[(mid, tables, seed)]


def evaluate(cfg, combo, tables, seeds, loader):
    """Mean METRICS of the detector with cfg updated by combo, over the tune split."""
    c = dict(cfg, **combo)
    pred = lambda d: hz.dead_ball.predict_labels(d, c, hz.IDSSE_CTX)
    pred.__name__ = "dead_ball_v1"
    _, summary = hz.run(pred, "tune", tables, seeds, loader=loader)
    return dict(combo, **{k: float(summary[k]["mean"]) for k in METRICS},
                spell_f1_min=float(summary["spell_f1_mean"]["min"]),
                spell_f1_max=float(summary["spell_f1_mean"]["max"]),
                n_pred_spells=float(summary["n_pred_spells"]["mean"]),
                n_gt_spells=float(summary["n_gt_spells"]["mean"]))


def _work(args):
    cfg, combo, tables, seeds = args
    return evaluate(cfg, combo, tables, seeds, _cached)


def run_parallel(cfg, combos, tables, seeds, workers):
    with cf.ProcessPoolExecutor(workers, initializer=_init, initargs=(tables, seeds)) as ex:
        return list(ex.map(_work, [(cfg, c, tables, seeds) for c in combos], chunksize=20))


def combos_of(grid):
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


def apriori_distance(row, grid, apriori):
    return sum(abs(grid[k].index(row[k]) - grid[k].index(apriori[k])) for k in grid)


def choose(results, grid, apriori):
    """TIE_RULE. results in grid order."""
    best = max(r["spell_f1_mean"] for r in results)
    near = [r for r in results if r["spell_f1_mean"] >= best - SPELL_TIE_BAND]
    top_ad = max(r["alive_dead_f1"] for r in near)
    near = [r for r in near if r["alive_dead_f1"] >= top_ad - 1e-12]
    return min(near, key=lambda r: apriori_distance(r, grid, apriori))   # min is first on ties


def interior(row, grid):
    return {k: 0 < grid[k].index(row[k]) < len(grid[k]) - 1 for k in grid}


def tune(log_path, workers, extend, previous):
    if os.path.exists(log_path):
        raise SystemExit("{} exists: refusing to overwrite a tune log".format(log_path))
    cfg = hz.load_config()
    grid = {k: sorted(set(v) | set(extend.get(k, ()))) for k, v in GRID.items()}
    apriori = {k: cfg[k] for k in grid}
    for k in grid:
        assert apriori[k] in grid[k], "a priori {}={} not in its grid".format(k, apriori[k])
    combos = combos_of(grid)
    head = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "grid": grid, "extended": extend, "previous_run": previous, "a_priori": apriori,
            "fixed": {k: cfg[k] for k in cfg if k.startswith("dead_") and k not in grid},
            "seeds": list(SEEDS), "tables": "degraded (ball_loss_variant a_priori)",
            "tune_ids": list(hz.split_ids("tune", False)),
            "held_out_ids_not_used": list(hz.ef.IDSSE_HOLDOUT),
            "objective": OBJECTIVE, "tie_rule": TIE_RULE,
            "script_sha256": hz.sha256(os.path.abspath(__file__)),
            "harness_sha256": hz.sha256(hz.__file__),
            "config_sha256": hz.sha256(hz.CONFIG_PATH), "source": hz.SOURCE}
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "x") as f:           # objective and tie rule on disk before any score
        json.dump(dict(head, status="running", n_combos=len(combos)), f, indent=1)
    results = run_parallel(cfg, combos, "degraded", SEEDS, workers)
    chosen = choose(results, grid, apriori)
    ad_best = max(results, key=lambda r: r["alive_dead_f1"])
    top10 = sorted(results, key=lambda r: -r["spell_f1_mean"])[:10]
    log = dict(head, status="done", n_combos=len(combos),
               chosen={k: chosen[k] for k in grid}, chosen_scores={m: chosen[m] for m in METRICS},
               chosen_interior=interior(chosen, grid), alive_dead_optimal=ad_best,
               top10_spell_f1_spread=top10[0]["spell_f1_mean"] - top10[-1]["spell_f1_mean"],
               results=results)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=1)
    print(json.dumps({k: log[k] for k in ("chosen", "chosen_scores", "chosen_interior",
                                          "alive_dead_optimal", "top10_spell_f1_spread")}, indent=1))


def choose_one(results, key):
    """ONE_KEY_TIE_RULE."""
    best = max(r["spell_f1_mean"] for r in results)
    return min((r for r in results if r["spell_f1_mean"] >= best - ONE_KEY_TIE_BAND),
               key=lambda r: r[key])


def tune_one(key, log_path, workers):
    """Grid one key (ONE_KEY_GRID) on degraded tuning tables; clean tables scored as a cross-check
    only (never used for selection). Log format compatible with frozen_from_log."""
    if os.path.exists(log_path):
        raise SystemExit("{} exists: refusing to overwrite a tune log".format(log_path))
    assert key in ONE_KEY_GRID, "--tune-one: {} not in ONE_KEY_GRID".format(key)
    cfg = hz.load_config()
    grid = {key: ONE_KEY_GRID[key]}
    combos = combos_of(grid)
    head = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "grid": grid,
            "fixed": {k: cfg[k] for k in cfg if k.startswith("dead_") and k != key},
            "seeds": list(SEEDS), "tables": "degraded (ball_loss_variant a_priori)",
            "tune_ids": list(hz.split_ids("tune", False)),
            "held_out_ids_not_used": list(hz.ef.IDSSE_HOLDOUT),
            "objective": OBJECTIVE, "tie_rule": ONE_KEY_TIE_RULE,
            "script_sha256": hz.sha256(os.path.abspath(__file__)),
            "harness_sha256": hz.sha256(hz.__file__),
            "detector_sha256": hz.sha256(hz.dead_ball.__file__),
            "config_sha256": hz.sha256(hz.CONFIG_PATH), "source": hz.SOURCE}
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "x") as f:           # objective and tie rule on disk before any score
        json.dump(dict(head, status="running", n_combos=len(combos)), f, indent=1)
    results = run_parallel(cfg, combos, "degraded", SEEDS, workers)
    chosen = choose_one(results, key)
    clean = run_parallel(cfg, combos, "clean", (), workers)
    log = dict(head, status="done", n_combos=len(combos), chosen={key: chosen[key]},
               chosen_scores={m: chosen[m] for m in METRICS + ("n_pred_spells",)}, results=results,
               clean_crosscheck_not_used_for_selection=clean)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=1)
    for name, rows in (("degraded", results), ("clean", clean)):
        print(name)
        for r in rows:
            print("  {:>4}  sf1 {:.4f} [{:.4f}-{:.4f}]  ad {:.4f}  spells {:.1f}{}".format(
                r[key], r["spell_f1_mean"], r["spell_f1_min"], r["spell_f1_max"],
                r["alive_dead_f1"], r["n_pred_spells"],
                "  <- chosen" if name == "degraded" and r[key] == chosen[key] else ""))
    print("data: " + hz.SOURCE)


def frozen_from_log(log_path, cfg, skip=()):
    """Config must equal the log's chosen row, except keys in skip (re-tuned by a later run).
    Returns the log's path, sha256 and timestamp."""
    if not os.path.isfile(log_path):
        raise SystemExit("{} missing: run --tune first".format(log_path))
    with open(log_path) as f:
        log = json.load(f)
    assert log["status"] == "done", "{}: tune not finished".format(log_path)
    diff = {k: (cfg[k], v) for k, v in dict(log["fixed"], **log["chosen"]).items()
            if k not in skip and cfg[k] != v}
    if diff:
        raise SystemExit("config differs from the tune log's chosen row (config, log): {}".format(diff))
    sha = hz.sha256(log_path)
    stale = {k: cfg["_provenance"][k].get("tune_log", {}).get("sha256") for k in log["chosen"] if k not in skip}
    stale = {k: v for k, v in stale.items() if v != sha}
    if stale:
        raise SystemExit("config provenance tune_log sha256 != {} ({}): {}".format(
            log_path, sha, stale))
    return {"path": os.path.relpath(log_path, hz.ROOT), "sha256": sha,
            "timestamp": log["timestamp"]}


def crosscheck(log_path, out_path, workers):
    """Chosen combo + 1-D profiles: degraded rows read from the log, clean tables scored afresh."""
    if os.path.exists(out_path):
        raise SystemExit("{} exists: refusing to overwrite".format(out_path))
    with open(log_path) as f:
        log = json.load(f)
    grid, chosen = log["grid"], log["chosen"]
    cfg = hz.load_config()
    deg = {tuple(r[k] for k in grid): r for r in log["results"]}
    profiles = {k: [dict(chosen, **{k: v}) for v in grid[k]] for k in grid}
    todo = [chosen] + [c for k in grid for c in profiles[k]]
    clean = run_parallel(cfg, todo, "clean", (), workers)
    cl = {tuple(r[k] for k in grid): r for r in clean}
    out = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "tune_log": log_path,
           "tune_log_sha256": hz.sha256(log_path), "script_sha256": hz.sha256(os.path.abspath(__file__)),
           "config_sha256": hz.sha256(hz.CONFIG_PATH), "chosen": chosen,
           "chosen_clean": cl[tuple(chosen.values())], "profiles": {}, "flags": [], "source": hz.SOURCE}
    for k in grid:
        rows = [{"value": c[k], "degraded": {m: deg[tuple(c.values())][m] for m in METRICS},
                 "clean": {m: cl[tuple(c.values())][m] for m in METRICS}} for c in profiles[k]]
        out["profiles"][k] = rows
        i_deg = grid[k].index(chosen[k])
        for m in ("spell_f1_mean", "alive_dead_f1"):
            cs = np.array([r["clean"][m] for r in rows])
            i_cl = int(np.argmax(cs))
            if abs(i_cl - i_deg) > 1:
                out["flags"].append("{}: clean {} optimum {} vs degraded choice {} (> 1 grid step)".format(
                    k, m, grid[k][i_cl], chosen[k]))
            if cs.max() - cs[i_deg] > 0.01:
                out["flags"].append("{}: clean {} at degraded choice {:.4f} is {:.4f} below the clean "
                                    "optimum {:.4f} at {}".format(k, m, cs[i_deg], cs.max() - cs[i_deg],
                                                                  cs.max(), grid[k][i_cl]))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "x") as f:
        json.dump(out, f, indent=1)
    for k, rows in out["profiles"].items():
        print(k)
        for r in rows:
            print("  {:>6}  deg sf1 {:.4f} ad {:.4f} | clean sf1 {:.4f} ad {:.4f}{}".format(
                r["value"], r["degraded"]["spell_f1_mean"], r["degraded"]["alive_dead_f1"],
                r["clean"]["spell_f1_mean"], r["clean"]["alive_dead_f1"],
                "  <- chosen" if r["value"] == chosen[k] else ""))
    print("flags:", out["flags"] or "none")
    print("data: " + hz.SOURCE)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tune", action="store_true")
    g.add_argument("--crosscheck", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--tune-one", metavar="KEY", help="one-key search over ONE_KEY_GRID[KEY]")
    p.add_argument("--log", required=True)
    p.add_argument("--out", help="--crosscheck output json")
    p.add_argument("--workers", type=int)
    p.add_argument("--extend", nargs="+", default=[], metavar="KEY=V1,V2",
                   help="extra grid values (edge extension run)")
    p.add_argument("--previous", help="path + sha256 of the run this one extends, logged")
    a = p.parse_args()
    if a.check:
        print(frozen_from_log(a.log, hz.load_config()))
        return
    if not a.workers:
        p.error("--workers required")
    if a.tune_one:
        tune_one(a.tune_one, a.log, a.workers)
    elif a.tune:
        ext = {}
        for e in a.extend:
            k, vs = e.split("=")
            assert k in GRID, "--extend: {} not a TUNABLE key".format(k)
            ext[k] = [type(GRID[k][0])(v) for v in vs.split(",")]
        prev = a.previous and {"path": a.previous, "sha256": hz.sha256(a.previous)}
        tune(a.log, a.workers, ext, prev)
    else:
        if not a.out:
            p.error("--out required with --crosscheck")
        crosscheck(a.log, a.out, a.workers)


if __name__ == "__main__":
    sys.exit(main())
