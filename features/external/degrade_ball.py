"""Simulate BAR-ATH-style ball loss on clean IDSSE tracking, then rebuild the features.

    python features/external/degrade_ball.py --tune --seeds 0 1          # 5 tuning matches only
    python features/external/degrade_ball.py --sensitivity --seeds 0 1   # reacq x onset table
    python features/external/degrade_ball.py --all --seeds 0 1 2         # write degraded tables
    python features/external/degrade_ball.py --all --seeds 0 1 2 \
        --ball-loss-params features/external/ball_loss_fitted_bar_ath.json  # fitted variant

Source: Bassek, Rein, Weber et al., "An integrated dataset of spatiotemporal and event data in
elite soccer", Sci Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y. Licence
CC BY 4.0. EXTERNAL data only; the provided BAR-ATH match never goes through this script.

IDSSE ball tracking is clean. BAR-ATH loses the ball in long blocks, mostly while it is dead, and
picks it up again at the restart (docs/findings.md). The model, per period, in this RNG order:
  1. each DFL dead span [s, e) maps to the gap [s + onset_lag, e + reacquisition_lag), clipped
     to the period. A gap of >= LONG_MIN_FRAMES is dropped with ball_loss_dead_drop_prob, a
     shorter non-empty one with ball_loss_short_gap_drop_prob (BAR-ATH has almost no 13-125
     frame runs: short stoppages lose the ball for > 5 s or not at all);
  2. live gaps: Poisson(rate * period hours) runs, log-uniform length in the frame range, placed
     uniformly inside live play with one free (alive, seen) frame on each side;
  3. flickers: same placement, uniform integer length in the frame range.
Each (match, seed) has its own stream, np.random.default_rng([seed, match_index]).
The active ball_loss_* values in config.json are a priori (2026-11-15). The earlier values fitted
by --tune to BAR-ATH's label-free tracking aggregates (DATA.md, docs/findings.md; never GT) are
kept, unused, in ball_loss_fitted_bar_ath.json and selected with --ball-loss-params, pending the
organisers' confirmation that such fitting is allowed. TARGET below is the --tune objective and
the comparison column of the printed tables; --tune and --sensitivity never write config.json.
With these assumptions, BOTH edges of every D spell that loses the ball on degraded IDSSE are
fixed by the simulator (onset lag and reacquisition lag), not observed.

Output external_data/idsse_degraded/<id>_s<seed>.parquet: the idsse_features columns (built by
idsse_features.match_features with ball_override, so the same extractor code), the clean `label`,
and `ball_dropped`; plus <id>_s<seed>_provenance.json. --tune writes tune_log.json there.
"""

import argparse
import concurrent.futures
import datetime
import itertools
import json
import os
import pathlib
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import idsse_features as idf  # noqa: E402
from idsse_features import ef  # noqa: E402
from fetch_idsse import EXPECTED_FPS, MATCHES  # noqa: E402
from diagnose_ball_gaps import DURATION_BINS_S, LONG_GAP_MIN_FRAMES as LONG_MIN_FRAMES  # noqa

SOURCE = ("Bassek, Rein, Weber et al., Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y, "
          "CC BY 4.0")
MATCH_INDEX = {m: i for i, m in enumerate(MATCHES)}
PARAM_KEYS = ("ball_loss_dead_drop_prob", "ball_loss_short_gap_drop_prob",
              "ball_loss_onset_lag_frames", "ball_loss_reacquisition_lag_frames",
              "ball_loss_live_gaps_per_hour", "ball_loss_live_gap_frames_range",
              "ball_loss_flickers_per_hour", "ball_loss_flicker_frames_range")
# BAR-ATH label-free tracking aggregates over its 158,857 graded frames (DATA.md, findings.md).
# Includes 2 long runs (540 frames) inside the 2nd-half pre-kickoff span, which IDSSE cannot
# produce (periods start at kickoff): a known small mismatch, kept.
BARATH_GRADED_FRAMES = 158857
TARGET = {"seen": 0.593, "long_runs": 132, "short_runs_1_12": 16, "mid_runs_13_125": 2,
          "long_median_s": 11.6, "long_max_s": 221.5, "long_share_of_missing": 0.997,
          "still_after_reappear": 0.02, "bins": [53, 35, 31, 11, 1, 1]}
COUNT_KEYS = ("long_runs", "short_runs_1_12", "mid_runs_13_125")
RUN_BINS_FRAMES = ((1, 12), (13, 25), (26, 125), (126, 10 ** 9))  # as ef.report_summary
GRID = {"ball_loss_dead_drop_prob": [0.85, 0.9, 0.95, 1.0],
        "ball_loss_short_gap_drop_prob": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
        "ball_loss_onset_lag_frames": [50, 75, 100, 125, 150],
        "ball_loss_live_gaps_per_hour": [15, 20, 25, 30, 35, 40],
        "ball_loss_live_gap_frames_range": [[126, 150], [126, 180], [126, 250], [126, 500]],
        "ball_loss_flickers_per_hour": [0.0, 3.0, 6.0, 9.0, 12.0]}


class NoRoom(Exception):
    """A run could not be placed: live play is too fragmented for the requested gaps."""


def true_runs(b):
    """Half-open (starts, stops) of the True runs of a 1-d bool array."""
    e = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def place_runs(mask, alive, lengths, rng):
    """Set mask on runs of the given lengths, each uniformly among starts whose run plus one frame
    on each side is alive and not yet masked. Raises NoRoom if a run does not fit."""
    for n in lengths:
        free = np.concatenate(([0], np.cumsum(alive & ~mask)))
        ok = free[n + 2:] - free[:-(n + 2)] == n + 2  # window [i-1, i+n+1) is all free
        starts = np.flatnonzero(ok) + 1
        if not len(starts):
            raise NoRoom("no room for a {}-frame run".format(n))
        i = starts[rng.integers(len(starts))]
        mask[i:i + n] = True


def check_params(p):
    for k in ("ball_loss_dead_drop_prob", "ball_loss_short_gap_drop_prob"):
        if not 0.0 <= p[k] <= 1.0:
            raise ValueError("{} = {} not in [0, 1]".format(k, p[k]))
    for k in ("ball_loss_onset_lag_frames", "ball_loss_reacquisition_lag_frames",
              "ball_loss_live_gaps_per_hour", "ball_loss_flickers_per_hour"):
        if p[k] < 0:
            raise ValueError("{} = {} < 0".format(k, p[k]))
    for k in ("ball_loss_live_gap_frames_range", "ball_loss_flicker_frames_range"):
        lo, hi = p[k]
        if not 1 <= lo <= hi:
            raise ValueError("{} = {} needs 1 <= lo <= hi".format(k, p[k]))


def drop_mask(alive, period, params, fps, rng):
    """Bool mask of rows whose ball is dropped. alive [N] 0/1 (DFL BallStatus), period [N]."""
    p = params
    check_params(p)
    (lg_lo, lg_hi), (fl_lo, fl_hi) = (p["ball_loss_live_gap_frames_range"],
                                      p["ball_loss_flicker_frames_range"])
    alive = alive.astype(bool)
    mask = np.zeros(len(alive), dtype=bool)
    for a, b in ef.block_spans(period):
        al, m = alive[a:b], mask[a:b]  # m is a view: writes go to mask
        starts, stops = true_runs(~al)
        g0 = starts + p["ball_loss_onset_lag_frames"]
        g1 = np.minimum(b - a, stops + p["ball_loss_reacquisition_lag_frames"])
        prob = np.where(g1 - g0 >= LONG_MIN_FRAMES, p["ball_loss_dead_drop_prob"],
                        np.where(g1 > g0, p["ball_loss_short_gap_drop_prob"], 0.0))
        keep = rng.random(len(starts)) < prob
        for s, e in zip(g0[keep], g1[keep]):
            m[s:e] = True
        hours = (b - a) / fps / 3600.0
        n_live = rng.poisson(p["ball_loss_live_gaps_per_hour"] * hours)
        place_runs(m, al, np.rint(np.exp(rng.uniform(np.log(lg_lo), np.log(lg_hi), n_live)))
                   .astype(int), rng)
        n_flick = rng.poisson(p["ball_loss_flickers_per_hour"] * hours)
        place_runs(m, al, rng.integers(fl_lo, fl_hi + 1, n_flick), rng)
    return mask


def match_rng(seed, mid):
    return np.random.default_rng([seed, MATCH_INDEX[mid]])


def missing_runs(missing, period):
    """Lengths (frames) of missing runs, per period so none joins across halftime."""
    out = []
    for a, b in ef.block_spans(period):
        s, e = true_runs(missing[a:b])
        out.append(e - s)
    return np.concatenate(out)


def still_after_reappear(mask, period, still_frac):
    """As diagnose_ball_gaps section 4: share of long gaps ending inside their period whose
    ball_still_frac 12 rows after the reappearance is >= 0.5. Only rows in the same period with a
    finite still_frac count; NaN if none."""
    vals = []
    for a, b in ef.block_spans(period):
        st, sp = true_runs(mask[a:b])
        idx = a + sp[(sp - st >= LONG_MIN_FRAMES) & (sp + 12 < b - a)] + 12
        v = still_frac[idx]
        vals.append(v[np.isfinite(v)])
    v = np.concatenate(vals)
    return float(np.mean(v >= 0.5)) if len(v) else np.nan


def aggregates(missing, period, fps):
    L = missing_runs(missing, period)
    long_s = L[L >= LONG_MIN_FRAMES] / fps
    return {"seen": 1.0 - missing.mean(), "long_runs": len(long_s),
            "short_runs_1_12": int((L <= 12).sum()),
            "mid_runs_13_125": int(((L > 12) & (L < LONG_MIN_FRAMES)).sum()),
            "long_median_s": float(np.median(long_s)) if len(long_s) else np.nan,
            "long_max_s": float(long_s.max()) if len(long_s) else np.nan,
            "long_share_of_missing": L[L >= LONG_MIN_FRAMES].sum() / max(1, L.sum()),
            "bins": np.histogram(long_s, DURATION_BINS_S)[0], "lengths": L}


def loss(agg, scale):
    """Tuning objective: chi-square-like distance (sum (sim - t)^2 / max(t, 1)) over the long-gap
    duration bins, the 1-12 and the 13-125 frame run counts (target counts scaled to this match's
    length), plus ((seen - 0.593) / 0.02)^2."""
    t = np.array(TARGET["bins"] + [TARGET["short_runs_1_12"], TARGET["mid_runs_13_125"]]) * scale
    sim = np.concatenate([agg["bins"], [agg["short_runs_1_12"], agg["mid_runs_13_125"]]])
    return (((sim - t) ** 2 / np.maximum(t, 1)).sum()
            + ((agg["seen"] - TARGET["seen"]) / 0.02) ** 2)


def print_runs(L):
    """Same missing-run histogram as extract_features.report_summary."""
    print("  ball-missing runs: {}".format(len(L)))
    for lo, hi in RUN_BINS_FRAMES:
        sel = (L >= lo) & (L <= hi)
        print("    {:>4}-{:<5} frames: {:>6} runs, {:>6.1%} of missing frames".format(
            lo, hi if hi < 10 ** 9 else "inf", int(sel.sum()), L[sel].sum() / max(1, L.sum())))


def load(mid, in_dir):
    d = np.load(os.path.join(in_dir, mid + ".npz"))
    return d["ball_alive"], d["period"].astype(np.int8), d["ball"], tuple(d["pitch_m"])


_W = {}


def _init(data, fixed, seeds, keys):
    _W.update(data=data, fixed=fixed, seeds=seeds, keys=keys)


def _score(combo):
    params = dict(_W["fixed"], **dict(zip(_W["keys"], combo)))
    tot = 0.0
    try:
        for m, (alive, period) in _W["data"].items():
            for s in _W["seeds"]:
                agg = aggregates(drop_mask(alive, period, params, EXPECTED_FPS, match_rng(s, m)),
                                 period, EXPECTED_FPS)
                tot += loss(agg, len(alive) / BARATH_GRADED_FRAMES)
    except NoRoom:  # infeasible combination: excluded, counted in the log
        return None
    return tot / (len(_W["data"]) * len(_W["seeds"]))


def tune(in_dir, cfg, seeds, grid, log_path, workers):
    """Full grid on the five tuning matches only (ef.IDSSE_TUNE). Writes every row to log_path."""
    print("tuning on {} only; seeds {}; grid {}".format(ef.IDSSE_TUNE, seeds, grid))
    data = {m: load(m, in_dir)[:2] for m in ef.IDSSE_TUNE}
    fixed = {k: cfg[k] for k in PARAM_KEYS if k not in grid}
    combos = list(itertools.product(*grid.values()))
    with concurrent.futures.ProcessPoolExecutor(workers, initializer=_init,
                                                initargs=(data, fixed, seeds, list(grid))) as ex:
        losses = list(ex.map(_score, combos, chunksize=max(1, len(combos) // (8 * workers))))
    if os.path.exists(log_path):  # the fitted variant cites this log; never overwrite it
        raise SystemExit("{} exists; move it aside before re-tuning".format(log_path))
    rows = sorted(({"loss": l, **dict(zip(grid, c))} for l, c in zip(losses, combos)),
                  key=lambda r: np.inf if r["loss"] is None else r["loss"])
    with open(log_path, "w") as f:
        json.dump({"timestamp": datetime.datetime.now().astimezone().isoformat(),
                   "tuned_on": list(ef.IDSSE_TUNE), "held_out_not_used": list(ef.IDSSE_HOLDOUT),
                   "seeds": seeds, "rng": "default_rng([seed, match_index])",
                   "match_index": MATCH_INDEX, "grid": grid, "fixed": fixed,
                   "objective": loss.__doc__, "target": TARGET, "source": SOURCE,
                   "script_sha256": idf.sha256(os.path.abspath(__file__)),
                   "infeasible": sum(r["loss"] is None for r in rows), "results": rows}, f, indent=1)
    print("  infeasible (no room): {} of {}; log {}".format(
        sum(r["loss"] is None for r in rows), len(rows), log_path))
    for r in rows[:10]:
        print("  {}".format(r))
    return rows


def sensitivity(in_dir, cfg, seeds):
    """Aggregates on the tuning matches for reacquisition lag x onset lag, other params fixed."""
    print("sensitivity on {} only; seeds {}".format(ef.IDSSE_TUNE, seeds))
    print("  reacq onset   seen  long  1-12 13-125 median_s  bins 5/10/20/40/80/160+   still")
    for reacq, onset in itertools.product([0, 12, 25], [50, 100, 150]):
        params = dict({k: cfg[k] for k in PARAM_KEYS}, ball_loss_reacquisition_lag_frames=reacq,
                      ball_loss_onset_lag_frames=onset)
        rows = []
        for m in ef.IDSSE_TUNE:
            alive, period, ball, pitch = load(m, in_dir)
            for s in seeds:
                mask = drop_mask(alive, period, params, EXPECTED_FPS, match_rng(s, m))
                b = ball.copy()
                b[mask] = np.nan
                sf = ef.ball_features(b, pitch, ef.block_spans(period), EXPECTED_FPS,
                                      cfg)["ball_still_frac"]
                agg = aggregates(mask, period, EXPECTED_FPS)
                agg["still_after_reappear"] = still_after_reappear(mask, period, sf)
                agg["scale"] = BARATH_GRADED_FRAMES / len(alive)
                rows.append(agg)
        mean = mean_row(rows)
        print("  {:>5} {:>5}  {:.3f} {:>5.1f} {:>5.1f} {:>6.1f} {:>8.1f}  {}  {:.3f}".format(
            reacq, onset, mean["seen"], mean["long_runs"], mean["short_runs_1_12"],
            mean["mid_runs_13_125"], mean["long_median_s"],
            "/".join("{:.0f}".format(x) for x in mean["bins"]), mean["still_after_reappear"]))


def mean_row(rows):
    """Mean over (match, seed); counts scaled to BAR-ATH length; NaN-aware."""
    out = {k: float(np.nanmean([r[k] * (r["scale"] if k in COUNT_KEYS else 1) for r in rows]))
           for k in TARGET if k != "bins"}
    out["bins"] = np.mean([r["bins"] * r["scale"] for r in rows], axis=0)
    return out


def table(rows, label):
    print("\n{} (mean over {} match-seeds; counts scaled to {} frames)".format(
        label, len(rows), BARATH_GRADED_FRAMES))
    mean = mean_row(rows)
    for k in TARGET:
        if k != "bins":
            print("  {:<24} {:>8.3f}   target {}".format(k, mean[k], TARGET[k]))
    print("  long-gap bins (s) {}: {}   target {}".format(
        DURATION_BINS_S[:-1], np.round(mean["bins"], 1).tolist(), TARGET["bins"]))


def with_ball_loss_params(cfg, path):
    """cfg with the 8 ball_loss_* values and their _provenance entries replaced by those in the
    JSON file at path (same shape as config.json). The file must hold exactly PARAM_KEYS."""
    with open(path) as f:
        var = json.load(f)
    got, prov = {k for k in var if not k.startswith("_")}, set(var["_provenance"])
    if got != set(PARAM_KEYS) or prov != set(PARAM_KEYS):
        raise ValueError("{} must hold exactly {} and their _provenance; has {} / {}".format(
            path, PARAM_KEYS, sorted(got), sorted(prov)))
    out = json.loads(json.dumps(cfg))
    out.update({k: var[k] for k in PARAM_KEYS})
    out["_provenance"].update(var["_provenance"])
    return out


def variant_stamp(cfg, params_path, log_path):
    """Provenance stamp for each output. If any active ball_loss key is tuned_external, the config
    must match the tune log (frozen_from_log); otherwise the tune log is not needed."""
    if any(cfg["_provenance"][k]["basis"] == "tuned_external" for k in PARAM_KEYS):
        return dict(frozen_from_log(log_path, cfg), ball_loss_variant=params_path or "config.json")
    return {"ball_loss_variant": params_path or "a_priori"}


def frozen_from_log(log_path, cfg):
    """Rule 2 audit trail: the config must equal the tune log's best row before any table (held-out
    ones included) is built. Returns the log's sha256 and timestamp for each output's provenance."""
    if not os.path.isfile(log_path):
        raise SystemExit("{} missing: run --tune before building tables".format(log_path))
    with open(log_path) as f:
        log = json.load(f)
    best = log["results"][0]
    diff = {k: (cfg[k], v) for k, v in dict(log["fixed"], **best).items()
            if k != "loss" and cfg[k] != v}
    if diff:
        raise SystemExit("config differs from the tune log's best row (config, log): {}".format(diff))
    return {"tune_log_sha256": idf.sha256(log_path), "tune_log_timestamp": log["timestamp"]}


def build(mid, seed, in_dir, out_dir, cfg, frozen):
    alive, period, ball, _ = load(mid, in_dir)
    params = {k: cfg[k] for k in PARAM_KEYS}
    mask = drop_mask(alive, period, params, EXPECTED_FPS, match_rng(seed, mid))
    ball = ball.copy()
    ball[mask] = np.nan
    df, info = idf.match_features(os.path.join(in_dir, mid + ".npz"),
                                  os.path.join(in_dir, mid + ".json"), cfg,
                                  cfg["kickoff_window_side"]["IDSSE"], ball_override=ball)
    assert (df["ball_seen"].values == ~mask).all(), "ball_seen must be exactly ~ball_dropped"
    df["ball_dropped"] = mask
    out = os.path.join(out_dir, "{}_s{}.parquet".format(mid, seed))
    df.to_parquet(out, index=False)
    prov = ef.provenance(cfg, info, out)
    prov["extractor_sha256"] = prov.pop("script_sha256")
    prov.update(script_sha256=idf.sha256(os.path.abspath(__file__)), seed=seed,
                rng="default_rng([{}, {}])".format(seed, MATCH_INDEX[mid]),
                degrade_params=params, source=SOURCE, built_at=datetime.datetime.now().astimezone().isoformat(),
                **frozen,
                input_npz_sha256=idf.sha256(os.path.join(in_dir, mid + ".npz")))
    with open(out[:-len(".parquet")] + "_provenance.json", "w") as f:
        json.dump(prov, f, indent=2, default=str)
    agg = aggregates(mask, period, EXPECTED_FPS)
    agg["still_after_reappear"] = still_after_reappear(mask, period,
                                                       df["ball_still_frac"].values)
    agg["scale"] = BARATH_GRADED_FRAMES / len(df)
    print("{} seed {}: ball seen {:.1%}, long gaps {}, median {:.1f} s, still after {:.0%}".format(
        mid, seed, agg["seen"], agg["long_runs"], agg["long_median_s"],
        agg["still_after_reappear"]))
    print_runs(agg["lengths"])
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matches", nargs="*", help="IDSSE match ids: " + " ".join(MATCHES))
    parser.add_argument("--all", action="store_true", help="all 7 matches")
    parser.add_argument("--tune", action="store_true", help="grid search on the 5 tuning matches")
    parser.add_argument("--sensitivity", action="store_true", help="reacq x onset table, tuning")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--config", default=os.path.join(ROOT, "features", "config.json"))
    parser.add_argument("--in-dir", default=os.path.join(ROOT, "external_data", "idsse"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "external_data", "idsse_degraded"))
    parser.add_argument("--ball-loss-params", default=None, metavar="FILE",
                        help="override the ball_loss_* keys and provenance (e.g. "
                             "features/external/ball_loss_fitted_bar_ath.json); none = config.json")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    if args.ball_loss_params:
        cfg = with_ball_loss_params(cfg, args.ball_loss_params)
    ef.check_provenance(cfg)
    print("ball-loss variant: {}".format(args.ball_loss_params or "config.json"))
    print("data: " + SOURCE)
    provided = str(pathlib.Path(ROOT, "data").resolve()).casefold()
    for d in (args.in_dir, args.out_dir):
        d = pathlib.Path(d).resolve()
        if provided in (str(q).casefold() for q in (d, *d.parents)):
            raise SystemExit("external data must not be under data/ (reserved for the provided match)")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.tune:
        tune(args.in_dir, cfg, args.seeds, GRID, os.path.join(args.out_dir, "tune_log.json"),
             args.workers)
        return
    if args.sensitivity:
        sensitivity(args.in_dir, cfg, args.seeds)
        return
    ids = list(MATCHES) if args.all else args.matches
    if not ids or set(ids) - set(MATCHES):
        parser.error("name match ids from {} or pass --all".format(list(MATCHES)))
    frozen = variant_stamp(cfg, args.ball_loss_params, os.path.join(args.out_dir, "tune_log.json"))
    res = {m: [build(m, s, args.in_dir, args.out_dir, cfg, frozen) for s in args.seeds]
           for m in ids}
    for label, group in (("TUNING matches", ef.IDSSE_TUNE),
                         ("HELD-OUT matches (evaluation only, never selected on)",
                          ef.IDSSE_HOLDOUT)):
        rows = [r for m in group if m in res for r in res[m]]
        if rows:
            table(rows, label)


if __name__ == "__main__":
    sys.exit(main())
