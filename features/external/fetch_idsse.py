"""Download the IDSSE matches (external data) and store each one compactly, with its labels.

    pip install -r features/external/requirements.txt
    python features/external/fetch_idsse.py --all             # all 7 matches, one at a time
    python features/external/fetch_idsse.py J03WPY J03WMX     # or just some

Source: Bassek, Rein, Weber et al., "An integrated dataset of spatiotemporal and event data in
elite soccer", Sci Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y. Licence
CC BY 4.0 -- credit the paper wherever this data is used (tech doc included). Downloaded through
kloppy from huggingface.co/datasets/pysport/idsse-data.

This is EXTERNAL data. Under the organisers' 2026-11-03 rules it is what parameters may be tuned
on. The provided BAR-ATH data never goes through this script.

Each match is loaded fully into memory (the raw position XML is large), converted, written, and
released before the next one starts, so peak memory is one match. The raw XML is downloaded into a
temporary folder that is deleted after conversion, so nothing raw is kept on disk.

Output, per match, in <repo>/external_data/idsse/ (gitignored). External data lives outside
data/ on purpose: data/ holds the provided match, which AI tools must never read, while external
data may be read by them freely.
  <match_id>.npz   frame_id [N] int32, period [N] int8,
                   players [N, S, 3] float32  (x_norm, y_norm, speed m/s), NaN when absent,
                   slot_team [S] int8         (0 home, 1 away),
                   ball [N, 3] float32        (x_norm, y_norm, z metres),
                   ball_alive [N] int8        (1 alive, 0 dead)      -- DFL BallStatus
                   ball_owner [N] int8        (0 home, 1 away, -1 none) -- DFL BallPossession
                   pitch_m [2] float32        (length, width) from the match information
  <match_id>.json  teams, slot player ids, counts, source, kloppy version

Coordinates are kloppy's normalised system (0-1 along length and width, origin top-left), the same
0-1 convention as the ATD, so the feature code can treat both the same way after scaling by
pitch_m. Only distances and the length axis are used downstream, so the y-axis direction does
not matter.
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

MATCHES = {  # kloppy's IDSSE ids -> fixture, as listed in kloppy._providers.sportec
    "J03WMX": "1. FC Koeln vs. FC Bayern Muenchen",
    "J03WN1": "VfL Bochum 1848 vs. Bayer 04 Leverkusen",
    "J03WPY": "Fortuna Duesseldorf vs. 1. FC Nuernberg",
    "J03WOH": "Fortuna Duesseldorf vs. SSV Jahn Regensburg",
    "J03WQQ": "Fortuna Duesseldorf vs. FC St. Pauli",
    "J03WOY": "Fortuna Duesseldorf vs. F.C. Hansa Rostock",
    "J03WR9": "Fortuna Duesseldorf vs. 1. FC Kaiserslautern",
}
EXPECTED_FPS = 25


def convert(dataset):
    """kloppy TrackingDataset -> dict of arrays (see module docstring). Fails on anything odd."""
    from kloppy.domain import BallState, Ground

    meta = dataset.metadata
    assert meta.frame_rate == EXPECTED_FPS, "frame rate {} != {}".format(
        meta.frame_rate, EXPECTED_FPS)
    teams = {t.ground: t for t in meta.teams}
    assert set(teams) == {Ground.HOME, Ground.AWAY}, "expected a home and an away team"
    home, away = teams[Ground.HOME], teams[Ground.AWAY]

    slot_players = list(home.players) + list(away.players)
    slot_team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)
    slot_of = {p.player_id: i for i, p in enumerate(slot_players)}
    assert len(slot_of) == len(slot_players), "duplicate player ids"

    frames = dataset.frames
    n, s = len(frames), len(slot_players)
    assert n > 0
    frame_id = np.empty(n, dtype=np.int32)
    period = np.empty(n, dtype=np.int8)
    players = np.full((n, s, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    ball_alive = np.empty(n, dtype=np.int8)
    ball_owner = np.full(n, -1, dtype=np.int8)

    for i, f in enumerate(frames):
        frame_id[i] = f.frame_id
        period[i] = f.period.id
        assert f.ball_state in (BallState.ALIVE, BallState.DEAD), \
            "frame {}: ball_state {!r}".format(f.frame_id, f.ball_state)
        ball_alive[i] = 1 if f.ball_state == BallState.ALIVE else 0
        if f.ball_owning_team is not None:
            assert f.ball_owning_team.team_id in (home.team_id, away.team_id)
            ball_owner[i] = 0 if f.ball_owning_team.team_id == home.team_id else 1
        if f.ball_coordinates is not None:
            b = f.ball_coordinates
            ball[i] = (b.x, b.y, b.z if b.z is not None else np.nan)
        for player, pdata in f.players_data.items():
            if pdata.coordinates is None:
                continue
            row = players[i, slot_of[player.player_id]]
            row[0], row[1] = pdata.coordinates.x, pdata.coordinates.y
            row[2] = pdata.speed if pdata.speed is not None else np.nan

    # Each period must be one contiguous, strictly increasing block of frame ids.
    for p in np.unique(period):
        ids = frame_id[period == p]
        assert np.all(np.diff(ids) == 1), "period {}: frame ids not contiguous".format(p)

    dims = meta.pitch_dimensions
    pitch_m = np.array([dims.pitch_length, dims.pitch_width], dtype=np.float32)
    assert np.all(np.isfinite(pitch_m)) and 90 <= pitch_m[0] <= 120 and 45 <= pitch_m[1] <= 90, \
        "implausible pitch {}".format(pitch_m)

    arrays = dict(frame_id=frame_id, period=period, players=players, slot_team=slot_team,
                  ball=ball, ball_alive=ball_alive, ball_owner=ball_owner, pitch_m=pitch_m)
    info = {
        "home": home.name, "away": away.name,
        "slot_player_ids": [p.player_id for p in slot_players],
        "n_frames": int(n), "n_slots": int(s),
        "periods": sorted(int(p) for p in np.unique(period)),
        "alive_share": float(ball_alive.mean()),
        "owner_share_when_alive": {
            "home": float(np.mean(ball_owner[ball_alive == 1] == 0)),
            "away": float(np.mean(ball_owner[ball_alive == 1] == 1)),
            "none": float(np.mean(ball_owner[ball_alive == 1] == -1)),
        },
        "ball_position_share": float(np.mean(~np.isnan(ball[:, 0]))),
    }
    return arrays, info


def fetch_one(match_id, out_dir):
    import kloppy
    from kloppy import sportec
    from kloppy.config import set_config

    npz = os.path.join(out_dir, match_id + ".npz")
    if os.path.isfile(npz):
        print("{}: already there, skipping".format(match_id))
        return
    print("{}: {} -- downloading and parsing (a few minutes)".format(match_id, MATCHES[match_id]))
    t0 = time.time()
    # kloppy caches downloads in ~/kloppy_cache by default, which would keep every raw XML on
    # disk. Point the cache at a temporary folder that is deleted once this match is converted.
    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".download-") as tmp:
        set_config("cache", tmp)
        dataset = sportec.load_open_tracking_data(match_id=match_id, coordinates="kloppy",
                                                  only_alive=False)
        arrays, info = convert(dataset)
        del dataset
    tmp = npz + ".partial.npz"  # write-then-rename so an interrupted run never looks complete
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz)
    info.update({"match_id": match_id, "fixture": MATCHES[match_id],
                 "source": "IDSSE (Bassek et al. 2025, Sci Data 12:195), CC BY 4.0, via kloppy",
                 "kloppy_version": kloppy.__version__})
    with open(os.path.join(out_dir, match_id + ".json"), "w") as f:
        json.dump(info, f, indent=2)
    print("  {} frames, alive {:.1%}, {:.0f} s, {:.0f} MB".format(
        info["n_frames"], info["alive_share"], time.time() - t0, os.path.getsize(npz) / 1e6))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matches", nargs="*", help="IDSSE match ids: " + " ".join(MATCHES))
    parser.add_argument("--all", action="store_true", help="fetch all 7 matches")
    parser.add_argument("--out-root", default=os.path.join(ROOT, "external_data"),
                        help="folder for external datasets (default <repo>/external_data)")
    args = parser.parse_args()
    ids = list(MATCHES) if args.all else args.matches
    if not ids:
        parser.error("name match ids or pass --all")
    unknown = [m for m in ids if m not in MATCHES]
    if unknown:
        parser.error("unknown match ids {}; have {}".format(unknown, list(MATCHES)))
    out_dir = os.path.join(args.out_root, "idsse")
    provided = os.path.realpath(os.path.join(ROOT, "data"))
    assert not os.path.realpath(out_dir).startswith(provided + os.sep), \
        "external data must not go under data/ (reserved for the provided match)"
    os.makedirs(out_dir, exist_ok=True)
    for m in ids:
        fetch_one(m, out_dir)


if __name__ == "__main__":
    sys.exit(main())
