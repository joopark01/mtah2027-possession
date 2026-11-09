"""Write a fake, full-length tactical bundle with KNOWN answers, to test extract_features.py.

    python features/tests/make_synthetic_match.py /tmp/synth_bundle
    MTAH_DATA_DIR=/tmp/synth_bundle python features/tests/test_synthetic.py

No real data is involved. The layout matches DATA.md (tactical/BAR-ATH_tactical_*), the raw line
format matches what starter/baseline.py parses: "<frame_id>:<x,y,s>;...;<x,y,s>:<x,y,z>".
Every scripted event below is asserted by test_synthetic.py; the constants are shared via EVENTS.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from games import GAMES, half_video_blocks  # noqa: E402

N = 199319
S_HOME, S_AWAY = 23, 23
LENGTH, WIDTH = GAMES["BAR-ATH"]["pitch"]
SEED = 0

EVENTS = {
    "line": (20000, 21000, 10.0),       # ball moves +x at 10 m/s on [20000, 21000)
    "short_gap": (20500, 20510),        # 10 missing frames inside the line -> interpolated
    "long_gap": (20600, 20650),         # 50 missing frames -> stays NaN
    "reverse": (21300, 21700, 21500, 8.0),  # 8 m/s, reverses direction at 21500
    "out": (22000, 22100, -0.02),       # ball at x_norm -0.02 -> 2.106 m outside
    "still": (23000, 23300),            # ball static
    "nearest": (24000, 24100, 1.0, 4.0),  # home player 1 m from ball, away 4 m
    "runner": (25000, 26000, 5.0),      # home slot 0 runs +x at 5 m/s
    "jump": (27000, 10.0),              # away slot 0 jumps -10 m for one frame (identity switch)
}


def write_meta(path):
    players = "".join(
        '<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name><ShirtNumber>{0}</ShirtNumber>'
        '</Player>'.format(i + 1, "T1" if i < S_HOME else "T2") for i in range(S_HOME + S_AWAY))
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?><Main><Metadata><Teams>'
                '<Team id="T1"><Name>FC Barcelona</Name></Team>'
                '<Team id="T2"><Name>Athletic Club</Name></Team></Teams>'
                '<Players>{}</Players></Metadata></Main>'.format(players))


def build_arrays():
    rng = np.random.default_rng(SEED)
    px = np.full((N, S_HOME + S_AWAY), np.nan); py = px.copy(); ps = px.copy()
    ball = np.full((N, 3), np.nan)
    halves = half_video_blocks("BAR-ATH")
    for half, first, last in halves:
        home_left = half == 1  # home defends low x in half 1, high x in half 2
        hx = rng.uniform(0.05, 0.45, 11); ax_ = rng.uniform(0.55, 0.95, 11)
        if not home_left:
            hx, ax_ = 1 - hx, 1 - ax_
        hy = rng.uniform(0.1, 0.9, 11); ay = rng.uniform(0.1, 0.9, 11)
        sl = slice(first - 30, last + 31)  # players present 30 frames before and after the half
        px[sl, :11] = hx; py[sl, :11] = hy
        px[sl, S_HOME:S_HOME + 11] = ax_; py[sl, S_HOME:S_HOME + 11] = ay
        ps[sl, :11] = 3.0; ps[sl, S_HOME:S_HOME + 11] = 3.0

    a, b, v = EVENTS["line"]
    t = np.arange(b - a) / 25.0
    ball[a:b, 0] = (20.0 + v * t) / LENGTH; ball[a:b, 1] = 0.5; ball[a:b, 2] = 0.0
    for key in ("short_gap", "long_gap"):
        g0, g1 = EVENTS[key]; ball[g0:g1] = np.nan

    a, b, turn, v = EVENTS["reverse"]
    xs = np.where(np.arange(a, b) < turn, 50.0 + v * (np.arange(a, b) - turn) / 25.0,
                  50.0 - v * (np.arange(a, b) - turn) / 25.0)
    ball[a:b, 0] = xs / LENGTH; ball[a:b, 1] = 0.3; ball[a:b, 2] = 0.0

    a, b, xo = EVENTS["out"]
    ball[a:b] = [xo, 0.5, 0.0]
    a, b = EVENTS["still"]
    ball[a:b] = [0.5, 0.5, 0.0]

    a, b, dh, da = EVENTS["nearest"]
    ball[a:b] = [0.5, 0.5, 0.0]
    px[a:b, :] = np.where(np.isnan(px[a:b, :]), np.nan, 0.9)  # push everyone away...
    px[a:b, 0] = 0.5 + dh / LENGTH; py[a:b, 0] = 0.5               # ...except one of each team
    px[a:b, S_HOME] = 0.5 - da / LENGTH; py[a:b, S_HOME] = 0.5

    a, b, v = EVENTS["runner"]
    px[a:b, 0] = (10.0 + v * np.arange(b - a) / 25.0) / LENGTH; py[a:b, 0] = 0.2
    ps[a:b, 0] = v  # raw speed field in m/s, consistent with the positions (speed-unit check)

    t, dx = EVENTS["jump"]
    px[t, S_HOME] -= dx / LENGTH
    return px, py, ps, ball


def write_raw(path, px, py, ps, ball):
    def grp(x, y, s):
        return "NaN,NaN,NaN" if np.isnan(x) else "{:.6f},{:.6f},{:.3f}".format(x, y, s)
    with open(path, "w") as f:
        for i in range(N):
            players = ";".join(grp(px[i, j], py[i, j], ps[i, j]) for j in range(px.shape[1]))
            f.write("{}:{}:{}\n".format(i + 1, players, grp(*ball[i])))


def write_tags(path):
    inst = [("BUILD UP", 100.0, 104.0, "Barcelona"), ("SET PIECES", 200.0, 205.0, "N/A"),
            ("PROGRESSION", 300.0, 310.0, "Atletic Club")]
    body = "".join("<instance><ID>{}</ID><start>{}</start><end>{}</end><code>{}</code>"
                   "<label><group>Team</group><text>{}</text></label></instance>".format(
                       i, s, e, c, t) for i, (c, s, e, t) in enumerate(inst))
    with open(path, "w") as f:
        f.write("<file><ALL_INSTANCES>{}</ALL_INSTANCES></file>".format(body))


if __name__ == "__main__":
    root = os.path.join(sys.argv[1], "tactical")
    os.makedirs(root, exist_ok=True)
    write_meta(os.path.join(root, "BAR-ATH_tactical_atd.xml"))
    write_raw(os.path.join(root, "BAR-ATH_tactical_atd.txt"), *build_arrays())
    write_tags(os.path.join(root, "BAR-ATH_tactical_smart_tagging.xml"))
    print("synthetic bundle in", root)
