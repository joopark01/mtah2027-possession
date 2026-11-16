"""Check extract_features.build() against the scripted answers in make_synthetic_match.py.

    python features/tests/make_synthetic_match.py /tmp/synth_bundle
    MTAH_DATA_DIR=/tmp/synth_bundle python features/tests/test_synthetic.py
"""

import json
import os
import pathlib
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# This test prints frame-level values, so it must never run on the provided match. Not an assert
# (stripped by -O); casefolded because macOS paths are case-insensitive. Unset -> KeyError;
# empty is refused too, since games.DATA_DIR falls back to ./data on an empty value.
_real = str(pathlib.Path(HERE, "..", "..", "data").resolve()).casefold()
_dir = os.environ["MTAH_DATA_DIR"]
if not _dir or str(pathlib.Path(_dir).resolve()).casefold() == _real:
    raise SystemExit("MTAH_DATA_DIR is empty or points at the real data/ dir; synthetic match only")
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import extract_features as ef  # noqa: E402
from make_synthetic_match import EVENTS, LENGTH  # noqa: E402

with open(os.path.join(os.path.dirname(HERE), "config.json")) as f:
    CFG = json.load(f)
K = CFG["velocity_half_window_frames"]


def close(name, got, want, tol):
    got = np.asarray(got, dtype=np.float64)
    assert np.all(np.abs(got - want) <= tol), "{}: got {} .. {}, want {} +/- {}".format(
        name, np.nanmin(got), np.nanmax(got), want, tol)
    print("  ok  {:<34} {}".format(name, want))


df, info = ef.build("BAR-ATH", None, CFG, use_cache=False)
assert len(df) == 199319
assert info["attack_sign_home"] == {1: 1, 2: -1}, info["attack_sign_home"]
print("  ok  attack signs", info["attack_sign_home"])

a, b, v = EVENTS["line"]
seg = df.iloc[a + 2 * K:20490]
close("ball_speed on straight line", seg["ball_speed"], v, 1e-3)
close("ball_accel on straight line", seg["ball_accel"], 0.0, 1e-2)
close("ball_turn_deg on straight line", seg["ball_turn_deg"], 0.0, 0.1)

g0, g1 = EVENTS["short_gap"]
gap = df.iloc[g0:g1]
assert gap["ball_interp"].all() and not gap["ball_seen"].any()
close("interpolated x in short gap", gap["ball_x_m"], 20.0 + v * (np.arange(g0, g1) - a) / 25.0, 1e-3)
close("gap length in short gap", gap["ball_gap_len"], g1 - g0, 0)
close("since_seen at gap start", df["ball_since_seen"].iloc[g0], 1, 0)

g0, g1 = EVENTS["long_gap"]
assert df["ball_x_m"].iloc[g0:g1].isna().all() and not df["ball_interp"].iloc[g0:g1].any()
print("  ok  long gap left NaN")

a, b, turn, v = EVENTS["reverse"]
close("turn angle at reversal", df["ball_turn_deg"].iloc[turn], 180.0, 1e-3)
close("speed before reversal", df["ball_speed"].iloc[turn - 50:turn - K], v, 1e-3)
close("accel peak at reversal (2v / (2k/fps))", df["ball_accel"].iloc[turn - K:turn + K + 1].max(),
      2 * v / (2.0 * K / 25.0), 1.0)

a, b, xo = EVENTS["out"]
close("ball_out_m when outside", df["ball_out_m"].iloc[a:b], -xo * LENGTH, 1e-3)
close("ball_out_m inside, centre width", df["ball_out_m"].iloc[23100], -34.0, 1e-3)

a, b = EVENTS["still"]
close("still_frac on static ball", df["ball_still_frac"].iloc[a + 20:b - 20], 1.0, 0)

a, b, dh, da = EVENTS["nearest"]
w = df.iloc[a + 5:b - 5]
close("h_dmin_ball_m", w["h_dmin_ball_m"], dh, 1e-3)
close("a_dmin_ball_m", w["a_dmin_ball_m"], da, 1e-3)
close("nearest_margin_m", w["nearest_margin_m"], da - dh, 1e-3)
assert (w["nearest_team"] == 0).all()
close("nearest_free_space_m", w["nearest_free_space_m"], dh + da, 1e-3)
close("h_n_within_3m", w["h_n_within_3m"], 1, 0)

a, b, v = EVENTS["runner"]
r = df.iloc[a + 10:b - 10]
close("home mean adv vx with one runner", r["h_adv_vx"], v / 11.0, 1e-3)
close("away mean adv vx", r["a_adv_vx"], 0.0, 1e-6)
assert (r["h_own_half_frac"] <= 1).all()

ht = df.iloc[100000]
assert ht["half"] == 0 and ht["n_tracked"] == 0 and np.isnan(ht["h_cx_m"])
print("  ok  halftime empty")

assert df["st_build_up_H"].iloc[2500:2600].all() and not df["st_build_up_H"].iloc[2600]
assert df["st_set_pieces_NA"].iloc[5000:5125].all()
assert df["st_progression_A"].iloc[7500:7750].all()
print("  ok  smart tagging intervals")
print("\nall synthetic checks passed")

# -- provenance guard: each broken config must be refused ------------------------------------------
import copy  # noqa: E402

def refused(mutate, label):
    bad = copy.deepcopy(CFG)
    mutate(bad)
    try:
        ef.check_provenance(bad)
    except ValueError:
        print("  ok  provenance refuses: {}".format(label))
        return
    raise AssertionError("check_provenance accepted a config with " + label)

ef.check_provenance(CFG)
refused(lambda c: c.update(new_threshold=1.0), "a key without provenance")
refused(lambda c: c["_provenance"].pop("fps"), "a missing entry")
refused(lambda c: c["_provenance"]["fps"].pop("source"), "a missing field")
refused(lambda c: c["_provenance"]["fps"].update(basis="vibes"), "an unknown basis")
refused(lambda c: c["_provenance"]["fps"].update(tuned_on="IDSSE"), "tuned_on without tuned_external")
refused(lambda c: c["_provenance"]["fps"].update(basis="tuned_external", tuned_on=None),
        "tuned_external without a dataset")
refused(lambda c: c["_provenance"]["fps"].update(basis="tuned_external", tuned_on="BAR-ATH public",
                                                    target="x"),
        "tuning on the provided game")
print("all provenance checks passed")

# -- tracked count is by position, not by speed --------------------------------------------------
close("n_tracked mid-half (11 + 11 scripted players)", df["n_tracked"].iloc[30000], 22, 0)
close("n_tracked_no_speed mid-half", df["n_tracked_no_speed"].iloc[30000], 0, 0)
print("all tracked-count checks passed")

# -- raw speed field unit check (report only) ----------------------------------------------------
sc = info["speed_unit_check"]
assert sc["n_samples"] > 900, sc  # the 5 m/s runner is the only moving player
close("speed-unit ratio, synthetic field in m/s", sc["ratio"], 1.0, 1e-2)
assert ef.speed_unit_verdict(sc["ratio"], CFG["speed_unit_check_tolerance"]) == "m/s"
assert ef.speed_unit_verdict(3.6 * sc["ratio"], CFG["speed_unit_check_tolerance"]) == "km/h"
assert ef.speed_unit_verdict(2.0, CFG["speed_unit_check_tolerance"]) == "unknown"
assert ef.speed_unit_verdict(float("nan"), CFG["speed_unit_check_tolerance"]) == "unknown"
# Half boundary: a window touching a different half (or half 0) must not contribute a sample.
h = np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int8)
one = np.full((len(h), 1), 5.0)
got = ef.speed_unit_ratio(one * 3.6, one, np.zeros_like(one), h, 1, 2.0)
assert got["n_samples"] == 6 and abs(got["ratio"] - 3.6) < 1e-12, got  # t = 3..6 and 9..10
got = ef.speed_unit_ratio(one, one * 0.3, np.zeros_like(one), h, 1, 2.0)
assert got["n_samples"] == 0 and np.isnan(got["ratio"]), got  # nobody moving
print("all speed-unit checks passed")

# -- speed features come from position velocity, not the raw speed field -------------------------
# Runner window: home slot 0 moves at v, every other player is static. The raw field says 3.0 for
# the static players and v for the runner, so raw-field features would read 35/11 and 0.
a, b, v = EVENTS["runner"]
r = df.iloc[a + 10:b - 10]
close("h_mean_speed, one runner (position)", r["h_mean_speed"], v / 11.0, 1e-3)
close("h_walking_frac, one runner", r["h_walking_frac"], 10 / 11.0, 1e-6)
close("a_mean_speed, all static", r["a_mean_speed"], 0.0, 1e-6)
close("a_walking_frac, all static", r["a_walking_frac"], 1.0, 0)
close("all_mean_speed, one runner", r["all_mean_speed"], v / 22.0, 1e-3)
close("all_walking_frac, one runner", r["all_walking_frac"], 21 / 22.0, 1e-6)
# One-frame 10 m jump of away slot 0 at t: window ends hit it at t-k and t+k (62.5 m/s there).
t, dx = EVENTS["jump"]
close("n_implausible_velocity at t-k, t+k", df["n_implausible_velocity"].iloc[[t - K, t + K]], 1, 0)
close("n_implausible_velocity elsewhere near jump",
      df["n_implausible_velocity"].iloc[np.r_[t - 3 * K:t - K, t - K + 1:t + K, t + K + 1:t + 3 * K]],
      0, 0)
close("a_mean_speed around jump (jump excluded)", df["a_mean_speed"].iloc[t - 3 * K:t + 3 * K], 0.0, 1e-6)
close("all_mean_speed around jump", df["all_mean_speed"].iloc[t - 3 * K:t + 3 * K], 0.0, 1e-6)
close("a_walking_frac around jump", df["a_walking_frac"].iloc[t - 3 * K:t + 3 * K], 1.0, 0)
# Half boundaries: players exist from first-30 to last+30, so without the same-half mask the first
# and last k frames of each half would get a finite (0) speed from a window leaving the half.
# adv_vx shares the masked velocity, so it must be NaN at exactly the same edges.
cols = ["h_mean_speed", "a_mean_speed", "all_mean_speed",
        "h_walking_frac", "a_walking_frac", "all_walking_frac", "h_adv_vx", "a_adv_vx"]
for hh, first, last in ef.half_video_blocks("BAR-ATH"):
    edge = np.r_[first - 5:first + K, last - K + 1:last + 3]
    assert df[cols].iloc[edge].isna().all().all(), "half {}: speed finite at a boundary".format(hh)
    assert df[cols].iloc[[first + K, last - K]].notna().all().all(), \
        "half {}: speed NaN one frame inside the boundary".format(hh)
    print("  ok  half {} speed NaN on [{}, {}) and [{}, {}], finite at {} and {}".format(
        hh, first - 5, first + K, last - K + 1, last + 2, first + K, last - K))
assert df[cols].iloc[100000].isna().all()  # halftime
got = ef.same_half_window(np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int8), 1)
assert got.tolist() == [False] * 3 + [True] * 4 + [False] * 2 + [True] * 2 + [False], got
print("all position-speed checks passed")
