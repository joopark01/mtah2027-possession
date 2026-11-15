"""IDSSE npz -> per-frame feature table with the BAR-ATH columns plus a `label` column.

    python features/external/idsse_features.py --all
    python features/external/idsse_features.py J03WPY J03WMX

Source: Bassek, Rein, Weber et al., "An integrated dataset of spatiotemporal and event data in
elite soccer", Sci Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y. Licence
CC BY 4.0. EXTERNAL data (fetched by fetch_idsse.py); the provided BAR-ATH match never goes
through this script.

Adaptation only: no degradation, no tuning. Features come from
extract_features.features_from_arrays, the same code as the provided match.

Input  external_data/idsse/<id>.npz + <id>.json (format: fetch_idsse.py docstring).
Output external_data/idsse_features/<id>.parquet and <id>_provenance.json.

Differences from the BAR-ATH table:
  * no st_ columns (IDSSE has no Smart Tagging);
  * every frame is in a period, so half = period and graded is True everywhere;
  * ball_raw_f2 is ball z in metres (DFL Z); on BAR-ATH the third ball field is undocumented;
  * label: "D" where DFL BallStatus is dead, else "H"/"A" from DFL BallPossession (0/1). DFL's
    alive/dead timing and possession hand-over are a PROXY for the organisers' GT convention;
  * `frame` is the row index; rows of the two periods are adjacent, so anything temporal
    (smoothing, decoding, spells) must split on `half`. `frame_id` keeps the DFL id.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "features"))
import extract_features as ef  # noqa: E402
from fetch_idsse import EXPECTED_FPS, FETCH_VERSION, MATCHES  # noqa: E402

LABELS = ["H", "A", "D"]
KLOPPY_ORIENTATION_SIGNS = {"Orientation.HOME_AWAY": {1: 1, 2: -1},
                            "Orientation.AWAY_HOME": {1: -1, 2: 1}}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def labels_from(ball_alive, ball_owner):
    assert set(np.unique(ball_alive)) <= {0, 1}, "ball_alive must be 0/1"
    assert set(np.unique(ball_owner)) <= {0, 1}, "ball_owner must be 0/1"
    lab = np.where(ball_alive == 0, "D", np.where(ball_owner == 0, "H", "A"))
    return pd.Categorical(lab, categories=LABELS)


def label_runs(codes):
    """Lengths of the maximal constant runs of a 1-d array."""
    cuts = np.concatenate(([0], np.flatnonzero(np.diff(codes)) + 1, [len(codes)]))
    return np.diff(cuts)


def match_features(npz_path, json_path, cfg, kickoff_side, ball_override=None):
    """One IDSSE match -> (DataFrame, info). kickoff_side is passed in so the test can force it.
    ball_override: optional [N, 3] ball array used instead of the npz ball (degrade_ball.py)."""
    d = dict(np.load(npz_path))
    if ball_override is not None:
        assert ball_override.shape == d["ball"].shape, "ball_override shape {} != {}".format(
            ball_override.shape, d["ball"].shape)
        d["ball"] = ball_override
    with open(json_path) as f:
        meta = json.load(f)
    assert meta["fetch_version"] == FETCH_VERSION, "{}: fetch_version {} != {}, re-fetch".format(
        npz_path, meta["fetch_version"], FETCH_VERSION)
    period = d["period"].astype(np.int8)
    assert set(np.unique(period)) == {1, 2}, "periods {}".format(np.unique(period))
    for p in (1, 2):
        first = int(np.flatnonzero(period == p)[0])
        assert int(d["frame_id"][first]) == meta["period_first_frame"][str(p)], (
            "period {}: frame_id at first row {} != json period_first_frame {}".format(
                p, int(d["frame_id"][first]), meta["period_first_frame"][str(p)]))
    # Kickoff = first alive row of the period (DFL BallStatus). A period can open with dead rows
    # (J03WOH period 2: 48, ball 51 m from the spot), so its first row is not the kickoff.
    kickoff_rows = {p: int(np.flatnonzero((period == p) & (d["ball_alive"] == 1))[0])
                    for p in (1, 2)}
    pitch = tuple(float(v) for v in d["pitch_m"])
    # reported, not asserted: DFL data can start a few tenths of a second after the kick
    ball_off_m = {p: float(np.hypot((d["ball"][k, 0] - 0.5) * pitch[0],
                                    (d["ball"][k, 1] - 0.5) * pitch[1]))
                  for p, k in kickoff_rows.items()}
    feats, signs, speed_check = ef.features_from_arrays(
        d["players"], d["slot_team"], d["ball"], period, pitch, EXPECTED_FPS, kickoff_rows,
        kickoff_side, cfg)
    # Wiring check, NOT independent: kloppy derives orientation from period 1's first frame with the
    # same centroid comparison, and period 2 is its flip (HOME_AWAY = home
    # attacks +x in period 1, AWAY_HOME = -x; kloppy.domain.AttackingDirection.from_orientation).
    expected = KLOPPY_ORIENTATION_SIGNS[meta["orientation"]]
    assert signs == expected, "attack signs {} disagree with kloppy {} ({})".format(
        signs, expected, meta["orientation"])
    df = pd.DataFrame(feats)
    floats = df.select_dtypes("float64").columns  # same storage cast as ef.build()
    df[floats] = df[floats].astype(np.float32)
    df["frame_id"] = d["frame_id"]  # DFL frame id; `frame` is only the row index here
    df["label"] = labels_from(d["ball_alive"], d["ball_owner"])
    return df, {"attack_sign_home": signs, "speed_unit_check": speed_check,
                "kickoff_rows": kickoff_rows, "kickoff_ball_from_centre_m": ball_off_m,
                "teams": (meta["home"], meta["away"])}


def run_one(mid, in_dir, out_dir, cfg):
    npz_path = os.path.join(in_dir, mid + ".npz")
    print("{} ({})".format(mid, MATCHES[mid]))
    df, info = match_features(npz_path, os.path.join(in_dir, mid + ".json"), cfg,
                              cfg["kickoff_window_side"]["IDSSE"])
    out = os.path.join(out_dir, mid + ".parquet")
    df.to_parquet(out, index=False)
    prov = ef.provenance(cfg, info, out)
    prov["extractor_sha256"] = prov.pop("script_sha256")
    prov.update(script_sha256=sha256(os.path.abspath(__file__)), input_npz=npz_path,
                input_npz_sha256=sha256(npz_path),
                source="Bassek, Rein, Weber et al., Sci Data 12, 195 (2025), "
                       "doi:10.1038/s41597-025-04505-y, CC BY 4.0")
    with open(os.path.join(out_dir, mid + "_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2, default=str)
    print("  kickoff rows {}  ball from centre spot there (m) {}".format(
        info["kickoff_rows"], {p: round(v, 1) for p, v in info["kickoff_ball_from_centre_m"].items()}))
    shares = df["label"].value_counts(normalize=True)
    print("  rows {}  cols {}  labels {}  signs {}  ball seen {:.1%}  speed ratio {:.3f} ({})".format(
        len(df), df.shape[1], " ".join("{} {:.1%}".format(k, shares[k]) for k in LABELS),
        info["attack_sign_home"], df["ball_seen"].mean(), info["speed_unit_check"]["ratio"],
        ef.speed_unit_verdict(info["speed_unit_check"]["ratio"],
                              cfg["speed_unit_check_tolerance"])))
    # Label run lengths within a period: the noise floor of the DFL proxy labels under spellF1.
    runs = np.concatenate([label_runs(g["label"].cat.codes.values) for _, g in df.groupby("half")])
    print("  label runs {}: <5 frames {}, 5-24 {}, >=25 {}".format(
        len(runs), int((runs < 5).sum()), int(((runs >= 5) & (runs < 25)).sum()),
        int((runs >= 25).sum())))
    z = df["ball_raw_f2"]
    print("  ball z (f2) m: p1 {:.2f}  median {:.2f}  p99 {:.2f}  max {:.2f}".format(
        z.quantile(0.01), z.median(), z.quantile(0.99), z.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matches", nargs="*", help="IDSSE match ids: " + " ".join(MATCHES))
    parser.add_argument("--all", action="store_true", help="all 7 matches")
    parser.add_argument("--config", default=os.path.join(ROOT, "features", "config.json"))
    parser.add_argument("--in-dir", default=os.path.join(ROOT, "external_data", "idsse"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "external_data", "idsse_features"))
    args = parser.parse_args()
    ids = list(MATCHES) if args.all else args.matches
    if not ids:
        parser.error("name match ids or pass --all")
    unknown = [m for m in ids if m not in MATCHES]
    if unknown:
        parser.error("unknown match ids {}; have {}".format(unknown, list(MATCHES)))
    # Same guard as fetch_idsse.py: not an assert (stripped by -O); casefolded for macOS.
    provided = str(pathlib.Path(ROOT, "data").resolve()).casefold()
    out = pathlib.Path(args.out_dir).resolve()
    if provided in (str(q).casefold() for q in (out, *out.parents)):
        raise SystemExit("external data must not go under data/ (reserved for the provided match)")
    with open(args.config) as f:
        cfg = json.load(f)
    ef.check_provenance(cfg)
    os.makedirs(args.out_dir, exist_ok=True)
    for m in ids:
        run_one(m, args.in_dir, args.out_dir, cfg)


if __name__ == "__main__":
    sys.exit(main())
