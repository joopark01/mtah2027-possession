"""Synthetic checks for the raw-attribute and speed-unit checks in features/external/fetch_idsse.py
(IDSSE data: Bassek, Rein, Weber et al., Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y,
CC BY 4.0). Uses only made-up arrays and an inline XML string, no real data.

    python features/tests/test_fetch_checks.py
"""

import io
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external"))
from fetch_idsse import (check_raw, check_speed_ratio, count_missing_raw, player_row,  # noqa: E402
                         speed_ratio)

FPS = 25
PITCH_M = np.array([105.0, 68.0])


def make_players(speed_scale):
    """Two periods of 100 frames, 2 players running straight at 3 and 5 m/s; S = speed * scale."""
    rng = np.random.default_rng(0)
    n = 200
    period = np.repeat([1, 2], n // 2).astype(np.int8)
    players = np.full((n, 2, 3), np.nan, dtype=np.float32)
    for j, v_mps in enumerate((3.0, 5.0)):
        x_m = 10 + v_mps * np.arange(n) / FPS + rng.normal(0, 0.02, n)
        players[:, j, 0] = x_m / PITCH_M[0]
        players[:, j, 1] = 0.5
        players[:, j, 2] = v_mps * speed_scale
    players[50:60, 1] = np.nan  # player off the pitch for a while
    players[100:, :, 0] += 0.3  # period 2 shifted: the jump at the boundary must be ignored
    return players, period


def raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


XML = """<PositionsData><Positions>
<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="DFL-OBJ-0000XT">
  <Frame N="10000" X="0.1" Y="0.2" Z="0.1" S="1.0" BallPossession="1" BallStatus="1"/>
  <Frame N="10001" X="0.1" Y="0.2" Z="0.1" S="1.0" {ball_extra}BallStatus="0"/>
</FrameSet>
<FrameSet GameSection="firstHalf" TeamId="T1" PersonId="P1">
  <Frame N="10000" X="1.0" Y="2.0" S="3.0"/>
  <Frame N="10001" X="1.0" Y="2.0" {player_extra}/>
</FrameSet>
</Positions></PositionsData>"""


def xml(ball_extra, player_extra):
    return io.BytesIO(XML.format(ball_extra=ball_extra, player_extra=player_extra).encode())


def main():
    p, per = make_players(1.0)
    r = speed_ratio(p, per, PITCH_M, FPS)
    assert abs(r - 1) < 0.02, r
    check_speed_ratio(r)
    p, per = make_players(3.6)
    r = speed_ratio(p, per, PITCH_M, FPS)
    assert abs(r - 3.6) < 0.1, r
    assert raises(check_speed_ratio, r), "km/h must fail"
    # km/h input through the real conversion (player_row, as convert() uses) passes the check;
    # this fails if the conversion is dropped (ratio 3.6) or applied twice (ratio 0.28).
    for i, j in np.ndindex(p.shape[:2]):
        x, y, s_kmh = p[i, j]
        p[i, j] = player_row(SimpleNamespace(coordinates=SimpleNamespace(x=x, y=y), speed=s_kmh))
    check_speed_ratio(speed_ratio(p, per, PITCH_M, FPS))
    assert abs(player_row(SimpleNamespace(coordinates=SimpleNamespace(x=0, y=0), speed=36.0))[2]
               - 10.0) < 1e-9

    # A pair of frames straddling the half must never be used: here it is the only moving pair,
    # so with the same-period mask there is nothing to check on (the shifted period 2 above has
    # one boundary pair per player, which the median alone would hide).
    two = np.array([[[0.1, 0.5, 5.0]], [[0.4, 0.5, 5.0]]], dtype=np.float32)
    try:
        speed_ratio(two, np.array([1, 2], dtype=np.int8), PITCH_M, FPS)
        raise RuntimeError("speed_ratio used a pair across the half")
    except AssertionError:
        pass

    check_raw(xml('BallPossession="2" ', 'S="3.0"'))
    c = count_missing_raw(xml("", 'S="3.0"'))
    assert c["ball BallPossession"] == 1 and sum(c.values()) == 1, c
    assert raises(check_raw, xml("", 'S="3.0"')), "missing BallPossession must fail"
    c = count_missing_raw(xml('BallPossession="2" ', ""))
    assert c["player S"] == 1 and sum(c.values()) == 1, c
    assert raises(check_raw, xml('BallPossession="2" ', "")), "missing S must fail"
    c = count_missing_raw(xml('BallPossession="3" ', 'S="3.0"'))
    assert c["ball BallPossession bad value"] == 1 and sum(c.values()) == 1, c
    assert raises(check_raw, xml('BallPossession="3" ', 'S="3.0"')), "BallPossession 3 must fail"
    bad_status = io.BytesIO(XML.format(ball_extra='BallPossession="2" ', player_extra='S="3.0"')
                            .replace('BallStatus="0"', 'BallStatus="2"').encode())
    c = count_missing_raw(bad_status)
    assert c["ball BallStatus bad value"] == 1 and sum(c.values()) == 1, c
    bad_status.seek(0)
    assert raises(check_raw, bad_status), "BallStatus 2 must fail"
    print("test_fetch_checks: all passed")


if __name__ == "__main__":
    main()
