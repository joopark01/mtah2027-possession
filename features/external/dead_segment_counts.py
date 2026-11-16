"""Aggregate D-segment counts for the tuned detector vs DFL D runs (external data only).

Tuning matches only. The one held-out run (2026-11-16) was made with an earlier version of this
script that also looped over IDSSE_HOLDOUT; per the held-out policy in CLAUDE.md that loop is gone.
"""
import json, sys
import numpy as np
sys.path[:0] = ["features", "features/external"]
import extract_features as ef, dead_ball as db, harness as hz
cfg = json.load(open("features/config.json"))
def druns(d, half):
    n = 0; lens = []
    for a, b in ef.block_spans(half):
        x = np.r_[0, d[a:b].astype(np.int8), 0]; e = np.diff(x)
        L = np.flatnonzero(e == -1) - np.flatnonzero(e == 1); lens += list(L)
    return np.array(lens)
for split, ids in (("tune", ef.IDSSE_TUNE),):
    for tables, seeds in (("degraded", (0, 1, 2)), ("clean", (None,))):
        r = []
        for m in ids:
            for s in seeds:
                df = hz.load_table(m, tables, s)
                half = df["half"].to_numpy()
                gt = druns(df["label"].to_numpy() == "D", half)
                out = db.detect(df.drop(columns=[c for c in hz.LEAK_COLUMNS if c in df]), cfg, hz.IDSSE_CTX)
                pr = druns(out["dead"], half)
                r.append((len(pr), len(gt), np.mean(pr < 25), np.mean(gt < 25), out["dead"].mean(), (df["label"] == "D").mean(), pr.min(), gt.min()))
        r = np.array(r, float)
        print("{:8} {:8} runs {:2}  D segs pred {:.1f} vs GT {:.1f} | <1 s: pred {:.1%} GT {:.1%} | D share pred {:.1%} GT {:.1%} | shortest pred {:.0f} GT {:.0f} frames".format(
            split, tables, len(r), *r[:, :6].mean(0), r[:, 6].min(), r[:, 7].min()))
