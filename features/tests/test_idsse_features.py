"""Synthetic check of features/external/idsse_features.py (IDSSE adapter; data: Bassek, Rein,
Weber et al., Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y, CC BY 4.0). Builds a tiny
made-up npz in a temp dir, no real data.

    .venv/bin/python features/tests/test_idsse_features.py
"""

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external"))
import idsse_features as idf  # noqa: E402
from idsse_features import ef  # noqa: E402

N_PER_PERIOD = 300
FPS = 25
FIRST_FRAME = {"1": 10000, "2": 100000}
DEAD_START = 5


def make_match(tmp, rng):
    """2 periods, 22 players; home on low x in period 1, high x in period 2, for the first second
    after each kickoff, then anywhere. Rows start exactly at the period-1 kickoff."""
    n = 2 * N_PER_PERIOD
    period = np.repeat([1, 2], N_PER_PERIOD).astype(np.int8)
    frame_id = np.concatenate([FIRST_FRAME["1"] + np.arange(N_PER_PERIOD),
                               FIRST_FRAME["2"] + np.arange(N_PER_PERIOD)]).astype(np.int32)
    slot_team = np.repeat([0, 1], 11).astype(np.int8)
    players = np.empty((n, 22, 3), dtype=np.float32)
    players[:, :, 0] = rng.uniform(0.05, 0.95, (n, 22))
    players[:, :, 1] = rng.uniform(0.05, 0.95, (n, 22))
    players[:, :, 2] = rng.uniform(0, 5, (n, 22))
    for p, home_low in ((1, True), (2, False)):
        k = (p - 1) * N_PER_PERIOD + (DEAD_START if p == 2 else 0)
        sl = slice(k, k + FPS + 1)
        low, high = rng.uniform(0.1, 0.4, (FPS + 1, 11)), rng.uniform(0.6, 0.9, (FPS + 1, 11))
        players[sl, :11, 0], players[sl, 11:, 0] = (low, high) if home_low else (high, low)
    ball = np.stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 2, n)], 1)
    alive = np.ones(n, dtype=np.int8)
    alive[100:140] = 0
    alive[N_PER_PERIOD:N_PER_PERIOD + DEAD_START] = 0  # period 2 opens dead: kickoff comes later
    alive[450:500] = 0
    owner = np.zeros(n, dtype=np.int8)
    owner[60:120] = 1  # switch inside a live span and carried into dead ball
    owner[380:470] = 1
    np.savez(os.path.join(tmp, "SYN.npz"), frame_id=frame_id, period=period, players=players,
             slot_team=slot_team, ball=ball.astype(np.float32), ball_alive=alive,
             ball_owner=owner, pitch_m=np.array([105.0, 68.0], dtype=np.float32))
    with open(os.path.join(tmp, "SYN.json"), "w") as f:
        json.dump({"home": "H", "away": "A", "period_first_frame": FIRST_FRAME,
                   "fetch_version": idf.FETCH_VERSION, "orientation": "Orientation.HOME_AWAY"}, f)
    expected = np.where(alive == 0, "D", np.where(owner == 0, "H", "A"))
    return expected, (players, slot_team, ball, period)


def main():
    with open(os.path.join(HERE, "..", "config.json")) as f:
        cfg = json.load(f)
    ef.check_provenance(cfg)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        expected, (players, team, ball, period) = make_match(tmp, rng)
        npz, js = os.path.join(tmp, "SYN.npz"), os.path.join(tmp, "SYN.json")
        df, info = idf.match_features(npz, js, cfg, cfg["kickoff_window_side"]["IDSSE"])
        assert info["attack_sign_home"] == {1: 1, 2: -1}, info["attack_sign_home"]
        assert info["kickoff_rows"] == {1: 0, 2: N_PER_PERIOD + DEAD_START}, info["kickoff_rows"]
        assert (df["label"].astype(str).values == expected).all(), "labels differ from script"
        assert set("HAD") == set(expected), "script must exercise all labels"
        feats, _, _ = ef.features_from_arrays(players, team, ball, period, (105.0, 68.0), FPS,
                                              {1: 0, 2: N_PER_PERIOD + DEAD_START}, "after", cfg)
        extra = {"label", "frame_id"}
        assert set(df.columns) == set(feats) | extra, set(df.columns) ^ (set(feats) | extra)
        # Period seam: rows N-1 and N are a halftime apart, so no window may span them.
        k, n1 = cfg["velocity_half_window_frames"], N_PER_PERIOD
        seam = np.arange(n1 - k, n1 + k)
        assert df["ball_speed"].iloc[seam].isna().all(), "ball speed spans the period seam"
        assert df["ball_speed"].iloc[[n1 - k - 1, n1 + k]].notna().all()
        assert df["ball_turn_deg"].iloc[seam].isna().all()
        assert df["ball_accel"].iloc[np.arange(n1 - 2 * k, n1 + 2 * k)].isna().all()
        # positive controls: the mask must not blank everything near the seam
        assert df["ball_turn_deg"].iloc[[n1 - k - 1, n1 + k]].notna().all()
        assert df["n_implausible_velocity"].iloc[n1 - k - 1] > 0
        assert (df["n_implausible_velocity"].iloc[seam] == 0).all(), "seam counted as jumps"
        assert (df["frame_id"].values[[0, n1]] == [10000, 100000]).all()
        with open(js) as f:
            meta = json.load(f)
        with open(js, "w") as f:
            json.dump(dict(meta, orientation="Orientation.AWAY_HOME"), f)
        try:
            idf.match_features(npz, js, cfg, "after")
        except AssertionError as e:
            assert "disagree with kloppy" in str(e), e
        else:
            raise AssertionError("signs vs kloppy orientation mismatch must fail")
        with open(js, "w") as f:
            json.dump(meta, f)
        try:
            idf.match_features(npz, js, cfg, "before")
        except AssertionError as e:
            assert "kickoff window leaves the array" in str(e), e
        else:
            raise AssertionError("side 'before' must fail: there are no rows before kickoff")
    # block_spans on edge cases, including a half that starts at row 0 and one that ends at N
    assert ef.block_spans(np.array([0, 0, 1, 1, 0, 2], dtype=np.int8)) == [
        (0, 2), (2, 4), (4, 5), (5, 6)]
    assert ef.block_spans(np.array([1, 1, 2, 2], dtype=np.int8)) == [(0, 2), (2, 4)]
    # ball_still_frac must not read across a block edge: still ball in block 0, fast in block 1
    n, h = 200, cfg["ball_still_half_window_frames"]
    b = np.zeros((n, 2)); b[:100] = 0.5
    b[100:, 0] = np.linspace(0, 1, 100); b[100:, 1] = 0.5
    half = np.repeat([1, 2], 100).astype(np.int8)
    bf = ef.ball_features(b, (105.0, 68.0), ef.block_spans(half), FPS, cfg)
    assert (bf["ball_still_frac"][100 - h:100] == 1.0).all(), "still_frac read the other block"
    # a held-out IDSSE match named as tuning data must be refused
    bad = json.loads(json.dumps(cfg))
    bad["_provenance"]["fps"].update(basis="tuned_external", tuned_on="IDSSE J03WMX,J03WPY")
    try:
        ef.check_provenance(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("check_provenance accepted a held-out match as tuning data")
    ok = json.loads(json.dumps(cfg))
    ok["_provenance"]["fps"].update(basis="tuned_external",
                                    tuned_on="IDSSE J03WMX; held out J03WPY,J03WR9",
                                    target="IDSSE DFL BallStatus")
    ef.check_provenance(ok)
    no_target = json.loads(json.dumps(ok))
    no_target["_provenance"]["fps"].pop("target")
    try:
        ef.check_provenance(no_target)
    except AssertionError:
        pass
    else:
        raise AssertionError("tuned_external without a target must be refused")
    # gap bookkeeping must not run across a block edge
    seen = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
    blocks = [(0, 5), (5, 9)]
    gap, since, until = (np.concatenate(p) for p in zip(
        *(ef.run_lengths_of_missing(seen[a:b]) for a, b in blocks)))
    bf = ef.ball_features(np.where(seen[:, None], 0.5, np.nan) * np.ones((9, 2)), (105.0, 68.0),
                          blocks, FPS, cfg)
    assert np.array_equal(bf["ball_gap_len"], gap, equal_nan=True)
    assert np.isnan(bf["ball_gap_len"][4]) and np.isnan(bf["ball_until_seen"][4]), "edge run"
    assert np.isnan(bf["ball_since_seen"][5]) and bf["ball_until_seen"][5] == 2
    assert bf["ball_gap_len"][1] == 2
    print("test_idsse_features: all checks passed")


if __name__ == "__main__":
    main()
