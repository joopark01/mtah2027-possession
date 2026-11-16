"""Synthetic check of features/external/tune_dead_ball.py (IDSSE data: Bassek, Rein, Weber et al.,
Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y, CC BY 4.0). Made-up tables through a
monkeypatched loader; no real files are read.

    .venv/bin/python features/tests/test_tune_dead_ball.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external"))
import tune_dead_ball as td  # noqa: E402

hz = td.hz


def table(n_half):
    """Two halves: live (running, ball seen) then a stoppage (ball missing, players walking)."""
    n = 2 * n_half
    live = np.arange(n) % n_half < n_half // 2
    lab = np.where(live, "H", "D")
    return pd.DataFrame({"half": np.repeat([1, 2], n_half).astype(np.int8),
                         "label": pd.Categorical(lab, categories=["H", "A", "D"]),
                         "ball_x_m": np.where(live, 0.0, np.nan), "ball_out_m": np.full(n, -5.0),
                         "ball_still_frac": np.zeros(n), "nearest_team": np.zeros(n, np.int8),
                         "h_walking_frac": np.where(live, 0.1, 1.0),
                         "a_walking_frac": np.where(live, 0.1, 1.0),
                         "all_walking_frac": np.where(live, 0.1, 1.0),
                         "all_mean_speed": np.where(live, 4.0, 0.5),
                         "ball_dropped": ~live})


def main():
    # codes(): same as pm.label_codes, element for element
    lab = np.array(["H", "D", "A", "X", "H"])
    assert (hz.codes(lab) == hz.pm.label_codes(lab)).all()

    # loader is only ever asked for tuning ids
    asked = []
    real_loader = hz.load_table
    hz.load_table = lambda mid, tables, seed: (asked.append(mid), table(400))[1]
    try:
        td._TABLES.clear()
        td._init("degraded", td.SEEDS)
        assert set(asked) == set(hz.ef.IDSSE_TUNE) and len(asked) == 5 * len(td.SEEDS), asked
        assert not set(asked) & set(hz.ef.IDSSE_HOLDOUT)

        # tune() end to end, in-process (spawned workers would not see the patch)
        def inproc(cfg, combos, tables, seeds, workers):
            td._TABLES.clear(); td._init(tables, seeds)
            return [td._work((cfg, c, tables, seeds)) for c in combos]
        td.run_parallel = inproc
        cfg0 = hz.load_config()
        small = {k: sorted({cfg0[k]} | set(v[1:3])) for k, v in td.GRID.items()}
        td.GRID.clear(); td.GRID.update(small)
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "tune_log.json")
            td.tune(log, 1, {}, None)
            out = json.load(open(log))
            assert out["tune_ids"] == list(hz.ef.IDSSE_TUNE) and len(out["results"]) == int(np.prod([len(v) for v in small.values()]))
            assert set(asked) == set(hz.ef.IDSSE_TUNE), "held-out table loaded"
            assert out["chosen"] == {k: out["chosen"][k] for k in td.GRID}
            try:
                td.tune(log, 1, {}, None)
                raise AssertionError("overwrote the log")
            except SystemExit:
                pass
            # freeze check
            cfg = dict(hz.load_config(), **out["chosen"])
            prov = {k: dict(v) for k, v in cfg["_provenance"].items()}
            for k in out["chosen"]:
                prov[k]["tune_log"] = {"sha256": hz.sha256(log)}
            cfg["_provenance"] = prov
            assert td.frozen_from_log(log, cfg)["timestamp"] == out["timestamp"]
            # same chosen row, different log file -> provenance sha mismatch
            other = os.path.join(tmp, "other_log.json")
            with open(other, "w") as f:
                json.dump(dict(out, timestamp="different"), f)
            try:
                td.frozen_from_log(other, cfg)
                raise AssertionError("freeze check passed a log the provenance does not name")
            except SystemExit as e:
                assert "sha256" in str(e), e
            bad = dict(cfg, dead_min_gap_frames=cfg["dead_min_gap_frames"] + 1)
            try:
                td.frozen_from_log(log, bad)
                raise AssertionError("freeze check passed a changed config")
            except SystemExit:
                pass
    finally:
        hz.load_table = real_loader

    # tie rule: within the band, highest alive_dead_f1, then closest to the a priori values
    grid = {"a": [0, 1, 2], "b": [0, 1, 2]}
    rs = [dict(a=0, b=0, spell_f1_mean=0.500, alive_dead_f1=0.9),
          dict(a=2, b=2, spell_f1_mean=0.499, alive_dead_f1=0.95),
          dict(a=1, b=1, spell_f1_mean=0.4985, alive_dead_f1=0.95),
          dict(a=2, b=0, spell_f1_mean=0.490, alive_dead_f1=0.99)]
    c = td.choose(rs, grid, {"a": 1, "b": 1})
    assert (c["a"], c["b"]) == (1, 1), c
    assert td.interior(c, grid) == {"a": True, "b": True}
    assert td.interior(rs[0], grid) == {"a": False, "b": False}
    print("test_tune_dead_ball: all checks passed")


if __name__ == "__main__":
    main()
