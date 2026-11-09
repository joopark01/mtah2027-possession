"""Download the IDSSE matches (external data) and store each one compactly, with its labels.

    pip install -r features/external/requirements.txt
    python features/external/fetch_idsse.py --all             # all 7 matches, one at a time
    python features/external/fetch_idsse.py J03WPY J03WMX     # or just some

Source: Bassek, Rein, Weber et al., "An integrated dataset of spatiotemporal and event data in
elite soccer", Sci Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y. Licence
CC BY 4.0 -- credit the paper wherever this data is used (tech doc included). Downloaded from
huggingface.co/datasets/pysport/idsse-data (URL pattern from kloppy) and parsed with kloppy.

This is EXTERNAL data. Under the organisers' 2026-11-03 rules it is what parameters may be tuned
on. The provided BAR-ATH data never goes through this script.

Each match is loaded fully into memory (the raw position XML is large), converted, written, and
released before the next one starts, so peak memory is one match. This script downloads the raw
XML itself into a temporary folder inside the output folder; the folder is deleted after
conversion (also on error), so nothing raw is kept on disk. SHA-256 and size of each raw file go
in the json.

kloppy silently fills missing attributes (BallPossession -> away, BallStatus -> dead, X/Y/Z/S ->
0), so before trusting it the raw positions XML is checked in one streaming pass: any ball frame
without BallPossession, BallStatus, X, Y or Z, or any player frame without X, Y or S, is an error,
and so is a BallPossession other than 1/2 or a BallStatus other than 0/1.

DFL speed S is km/h in the raw XML and is converted to m/s here (KMH_TO_MPS); the converted
speed is checked against the speed from displacement (ratios recorded in the json).

Output, per match, in <repo>/external_data/idsse/ (gitignored). External data lives outside
data/ on purpose: data/ holds the provided match, which AI tools must never read, while external
data may be read by them freely.
  <match_id>.npz   frame_id [N] int32, period [N] int8,
                   players [N, S, 3] float32  (x_norm, y_norm, speed m/s, converted from
                                              DFL S in km/h), NaN when the player is not
                                              on the pitch,
                   slot_team [S] int8         (0 home, 1 away),
                   ball [N, 3] float32        (x_norm, y_norm, z metres), always present,
                   ball_alive [N] int8        (1 alive, 0 dead)      -- DFL BallStatus
                   ball_owner [N] int8        (0 home, 1 away; never none; carried through dead
                                              ball (DFL convention)) -- DFL BallPossession
                   pitch_m [2] float32        (length, width) from the match information
  <match_id>.json  teams, slot player ids, counts, source, kloppy version, fetch_version, raw
                   file SHA-256 and bytes, speed ratios (raw km/h and converted m/s),
                   period_first_frame {1, 2} (kickoff frame ids, for attack_signs() in
                   features/extract_features.py) and kloppy's orientation.
                   Attacking direction is deliberately not inferred here.

Coordinates are kloppy's normalised system (0-1 along length and width, origin top-left), the same
0-1 convention as the ATD, so the feature code can treat both the same way after scaling by
pitch_m. Only distances and the length axis are used downstream, so the y-axis direction does
not matter.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time
import urllib.request

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
FETCH_VERSION = 2  # bump this whenever the output format or meaning changes (forces a re-fetch)
EXPECTED_FPS = 25
KMH_TO_MPS = 1 / 3.6  # DFL S is km/h; verified on J03WPY (raw S / displacement speed = 3.589)
SPEED_RATIO_TOL = 0.10  # S / displacement speed must be within 10% of 1 (m/s); a priori
MOVING_MPS = 2.0  # displacement speed above which a player counts as moving; a priori, well above
#                   per-frame position noise (a few cm at 25 fps is < 1 m/s)


def count_missing_raw(xml_source):
    """One streaming pass over a DFL positions XML (path or file object). Returns counts of ball
    frames missing BallPossession / BallStatus / X / Y / Z, of ball frames whose BallPossession is
    present but not "1"/"2" or BallStatus present but not "0"/"1" (kloppy maps anything but "1" to
    away / dead), and of player frames missing X / Y / S. Every non-ball FrameSet counts as a
    player (officials included, kloppy drops them later)."""
    from lxml import etree

    need = {"ball": ("BallPossession", "BallStatus", "X", "Y", "Z"), "player": ("X", "Y", "S")}
    allowed = {"BallPossession": ("1", "2"), "BallStatus": ("0", "1")}
    counts = {"{} {}".format(k, a): 0 for k, attrs in need.items() for a in attrs}
    counts.update({"ball {} bad value".format(a): 0 for a in allowed})
    kind = None
    for event, elem in etree.iterparse(xml_source, events=("start", "end"), huge_tree=True):
        if event == "start" and elem.tag == "FrameSet":
            kind = "ball" if elem.get("TeamId") == "BALL" else "player"
        elif event == "start" and elem.tag == "Frame":
            assert kind is not None, "Frame outside a FrameSet"
            for a in need[kind]:
                v = elem.get(a)
                if v is None:
                    counts["{} {}".format(kind, a)] += 1
                elif kind == "ball" and a in allowed and v not in allowed[a]:
                    counts["ball {} bad value".format(a)] += 1
        elif event == "end" and elem.tag in ("Frame", "FrameSet"):
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    return counts


def check_raw(xml_source):
    """Raise if any frame lacks an attribute kloppy would silently default, or has an out-of-range
    BallPossession / BallStatus (see count_missing_raw)."""
    counts = count_missing_raw(xml_source)
    bad = {k: v for k, v in counts.items() if v > 0}
    if bad:
        raise ValueError("raw positions XML has frames with missing or out-of-range attributes "
                         "kloppy would silently default: {}".format(bad))


def speed_ratio(players, period, pitch_m, fps):
    """Median of S / (|delta position| m per frame * fps) over moving players, using only frames
    where the player is present in both and both frames are in the same period. players [N, S, 3]
    as in the npz (x_norm, y_norm, speed m/s, already converted); frame ids must already be
    contiguous within a period."""
    xy_m = players[:, :, :2].astype(np.float64) * np.asarray(pitch_m, dtype=np.float64)
    disp_mps = np.linalg.norm(np.diff(xy_m, axis=0), axis=2) * fps  # [N-1, S], NaN if absent
    s_mid = 0.5 * (players[1:, :, 2] + players[:-1, :, 2]).astype(np.float64)
    same_period = (period[1:] == period[:-1])[:, None]
    use = same_period & np.isfinite(disp_mps) & np.isfinite(s_mid) & (disp_mps > MOVING_MPS)
    assert use.sum() > 0, "no moving-player samples to check the speed unit on"
    return float(np.median(s_mid[use] / disp_mps[use]))


def check_speed_ratio(ratio):
    """Raise unless the converted speed (m/s) / displacement speed ratio is within tolerance of 1."""
    if not abs(ratio - 1) <= SPEED_RATIO_TOL:
        raise ValueError("converted speed / displacement speed = {:.3f} (raw S / displacement = "
                         "{:.3f}), not within {:.0%} of 1: the km/h -> m/s conversion does not fit "
                         "this data. A raw ratio near 1 means S was already m/s; anything else "
                         "means S has another unit or the positions/pitch_m/fps are off."
                         .format(ratio, ratio / KMH_TO_MPS, SPEED_RATIO_TOL))


def player_row(pdata):
    """kloppy PlayerData -> (x_norm, y_norm, speed m/s). DFL S is km/h, converted here."""
    speed_mps = pdata.speed * KMH_TO_MPS if pdata.speed is not None else np.nan
    return pdata.coordinates.x, pdata.coordinates.y, speed_mps


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
    ball_owner = np.empty(n, dtype=np.int8)

    for i, f in enumerate(frames):
        frame_id[i] = f.frame_id
        period[i] = f.period.id
        assert f.ball_state in (BallState.ALIVE, BallState.DEAD), \
            "frame {}: ball_state {!r}".format(f.frame_id, f.ball_state)
        ball_alive[i] = 1 if f.ball_state == BallState.ALIVE else 0
        assert f.ball_owning_team is not None, "frame {}: no ball owner".format(f.frame_id)
        assert f.ball_owning_team.team_id in (home.team_id, away.team_id)
        ball_owner[i] = 0 if f.ball_owning_team.team_id == home.team_id else 1
        b = f.ball_coordinates
        ball[i] = (b.x, b.y, b.z)
        for player, pdata in f.players_data.items():
            if pdata.coordinates is None:
                continue
            players[i, slot_of[player.player_id]] = player_row(pdata)

    assert np.all(np.isfinite(ball)), "ball position missing in some frames"
    periods = sorted(int(p) for p in np.unique(period))
    assert periods == [1, 2], "periods {} != [1, 2]".format(periods)
    # Each period must be one contiguous, strictly increasing block of frame ids.
    for p in periods:
        ids = frame_id[period == p]
        assert np.all(np.diff(ids) == 1), "period {}: frame ids not contiguous".format(p)

    dims = meta.pitch_dimensions
    pitch_m = np.array([dims.pitch_length, dims.pitch_width], dtype=np.float32)
    assert np.all(np.isfinite(pitch_m)) and 90 <= pitch_m[0] <= 120 and 45 <= pitch_m[1] <= 90, \
        "implausible pitch {}".format(pitch_m)
    ratio = speed_ratio(players, period, pitch_m, EXPECTED_FPS)
    check_speed_ratio(ratio)

    arrays = dict(frame_id=frame_id, period=period, players=players, slot_team=slot_team,
                  ball=ball, ball_alive=ball_alive, ball_owner=ball_owner, pitch_m=pitch_m)
    info = {
        "home": home.name, "away": away.name,
        "slot_player_ids": [p.player_id for p in slot_players],
        "n_frames": int(n), "n_slots": int(s),
        "periods": periods,
        "period_first_frame": {p: int(frame_id[period == p][0]) for p in periods},
        "orientation": str(meta.orientation),
        "speed_ratio_S_over_displacement": ratio,  # after km/h -> m/s, expect ~1
        "speed_ratio_raw_S_kmh_over_displacement": ratio / KMH_TO_MPS,  # median scales linearly
        "alive_share": float(ball_alive.mean()),
        "owner_share_when_alive": {
            "home": float(np.mean(ball_owner[ball_alive == 1] == 0)),
            "away": float(np.mean(ball_owner[ball_alive == 1] == 1)),
        },
    }
    return arrays, info


def download(url, path):
    """Stream url to path; return its SHA-256 and size."""
    sha = hashlib.sha256()
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        assert r.status == 200, "{}: HTTP {}".format(url, r.status)
        for chunk in iter(lambda: r.read(1 << 20), b""):
            sha.update(chunk)
            f.write(chunk)
    return {"url": url, "sha256": sha.hexdigest(), "bytes": os.path.getsize(path)}


def fetch_one(match_id, out_dir):
    import kloppy
    from kloppy import sportec
    from kloppy._providers.sportec import get_IDSSE_url

    npz = os.path.join(out_dir, match_id + ".npz")
    js = os.path.join(out_dir, match_id + ".json")
    if os.path.isfile(npz) and os.path.isfile(js):
        with open(js) as f:
            version = json.load(f).get("fetch_version")
        if version == FETCH_VERSION:
            print("{}: already there, skipping".format(match_id))
            return
        print("{}: fetch_version {} != {}, stale -- re-fetching".format(
            match_id, version, FETCH_VERSION))
    print("{}: {} -- downloading and parsing (a few minutes)".format(match_id, MATCHES[match_id]))
    t0 = time.time()
    # Raw XML lives only in this folder, which is removed on exit, error or not.
    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".download-") as tmp:
        meta_xml = pathlib.Path(tmp, "matchinformation.xml")
        pos_xml = pathlib.Path(tmp, "positions.xml")
        raw_files = {"meta": download(get_IDSSE_url(match_id, "meta"), meta_xml),
                     "tracking": download(get_IDSSE_url(match_id, "tracking"), pos_xml)}
        check_raw(str(pos_xml))
        dataset = sportec.load_tracking(meta_data=meta_xml, raw_data=pos_xml,
                                        coordinates="kloppy", only_alive=False)
        arrays, info = convert(dataset)
        del dataset
    info.update({"match_id": match_id, "fixture": MATCHES[match_id],
                 "source": "IDSSE (Bassek et al. 2025, Sci Data 12:195, "
                           "doi:10.1038/s41597-025-04505-y), CC BY 4.0, parsed with kloppy",
                 "kloppy_version": kloppy.__version__, "fetch_version": FETCH_VERSION,
                 "raw_files": raw_files})
    # write-then-rename, npz then json: the json (with fetch_version) is the completion marker,
    # so an interrupted run never leaves a current json next to an old or missing npz
    tmp = npz + ".partial.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz)
    with open(js + ".partial", "w") as f:
        json.dump(info, f, indent=2)
    os.replace(js + ".partial", js)
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
    # Not an assert (stripped by -O); casefolded because macOS paths are case-insensitive.
    provided = str(pathlib.Path(ROOT, "data").resolve()).casefold()
    out = pathlib.Path(out_dir).resolve()
    if provided in (str(q).casefold() for q in (out, *out.parents)):
        raise SystemExit("external data must not go under data/ (reserved for the provided match)")
    os.makedirs(out_dir, exist_ok=True)
    for m in ids:
        fetch_one(m, out_dir)


if __name__ == "__main__":
    sys.exit(main())
