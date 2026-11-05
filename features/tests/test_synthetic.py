"""Check extract_features.build() against the scripted answers in make_synthetic_match.py.

    python features/tests/make_synthetic_match.py /tmp/synth_bundle
    MTAH_DATA_DIR=/tmp/synth_bundle python features/tests/test_synthetic.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
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
    except AssertionError:
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
refused(lambda c: c["_provenance"]["fps"].update(basis="tuned_external", tuned_on="BAR-ATH public"),
        "tuning on the provided game")
print("all provenance checks passed")
