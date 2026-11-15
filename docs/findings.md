# Findings from label-free checks on BAR-ATH

All of these come from the provided tracking and Smart Tagging only, with no ground truth and no video. They are allowed under the 2026-11-03 rules as automatic checks, and none of them is used to tune anything.

## 2026-11-09: feature extractor sanity (`extract_features.py --report`)
- **Parsing is correct.** Ball seen 59.3% and mean tracked 20.49 both match DATA.md.
- **Missing speeds:** 1.63 tracked slots per frame have a position but no speed. Speed features should use velocity computed from positions, not the raw speed field.
- **Nearest player** (when the ball position is known): home 56.4%, away 43.6%.
- **Ball outside the pitch:** 2.5% of graded frames.
- **Speeds:** ball median 5.1 m/s, p99 24.8; player median 1.47 m/s (from position velocity; 1.52 from the raw speed field, superseded 2026-11-09).
- **`ball_raw_f2` is undocumented.** Range 0–7.3, median 1.31. Don't use it until confirmed. (For comparison, IDSSE's ball z is in metres.)
- **The ball vanishes in long blocks, not flickers.** There are 150 missing runs; 132 of them are over 5 s and hold 99.7% of missing frames. Gap interpolation fills only 51 frames.

## 2026-11-09: what happens while the ball is missing (`diagnose_ball_gaps.py`)
132 long gaps cover 64,446 frames (40.6% of graded). Median length 11.6 s, longest 221.5 s.

Player speed and walking rows use position velocity (jump-filtered, masked at half edges), re-run 2026-11-09. The raw-speed-field version gave 0.95 / 0.60 walking and 1.04 / 2.00 m/s.

| | gap | ball seen |
|---|---|---|
| share of players walking (median) | **0.95** | 0.62 |
| mean player speed (median) | **1.00 m/s** | 1.93 m/s |
| inside any set-piece tag | 21.7% | 31.7% |
| inside a `goals` tag (H + A) | **22.6%** | 2.1% |

- **69 of 132 gaps** have a set-piece tag starting inside them or within 1 s after.
- **Where the ball was last seen:** in a penalty area 26% of the time (2× the 13% base rate), outside the pitch 16%. Speed when last seen is low (median 2.2 m/s).
- **Where it came back:** outside the pitch 21%, in a box 29%, at the centre spot 5%. The ball is still over the next second in **only 2%** of cases, and it moved a median of 11 m while missing.

**Interpretation:** the long gaps are overwhelmingly stoppages. The tracker loses the ball while it is dead and picks it up again once it is played, so **reappearance ≈ the restart**. A minority of gaps (lower quartile of walking share, 0.86) may still be live play.

**Consequence:** build the dead-ball detector before the broadcast alignment.

## 2026-11-15: where team walking share is NaN (`diagnose_ball_gaps.py` section 5)
Re-run on 2026-11-15; the numbers above reproduced exactly.
- `h_walking_frac` is NaN on 755 graded frames (0.48%). Of these, 74.8% are inside long ball gaps, 24.6% are ball-seen frames and 0.5% are short gaps. There are 4 runs; the longest is 30.0 s and the median 0.08 s.
- `a_walking_frac` is NaN on 8 frames (4 runs, longest 0.1 s).
- **Consequence:** `a_` has 4 runs of 2 frames each, which matches the half edges (±k masking). `h_` also has 4 runs, so its 30 s run probably joins one of those edges and is mostly inside a stoppage (likely the 2nd-half pre-kickoff stretch from 110808 to 113742; not checked further). The dead-ball detector must treat a NaN walking share as unknown, not as "not walking", and fall back to `all_walking_frac`.

# Findings on IDSSE (external, Bassek et al. 2025, CC BY 4.0)

## 2026-11-15: adapter (`features/external/idsse_features.py`)
- **Kickoff = first alive row of the period, not its first row.** Periods can open with dead rows (J03WOH period 2: 48 rows, ball 51 m from the spot). At the first alive row the ball is 0.2–10.1 m from the centre spot, because DFL data can start just after the kick. The attack signs agree with kloppy's orientation on all 7 matches, and this is asserted. That is a wiring check, not independent confirmation: kloppy uses the same centroid comparison on period 1's first frame. The kickoff rule is a data-format decision checked on all 7 matches, not a parameter selected on the held-out pair.
- **D share is 41–52%** (DFL BallStatus). Label runs: 462–614 per match, of which 5–16 are shorter than 5 frames. This is the flicker floor of the proxy labels.
- **Ball z (`ball_raw_f2`) on IDSSE is metres:** median 0.0–0.09, p99 about 9, max 27. Do not compare it with BAR-ATH's f2, which is undocumented.
- **Proxy mismatch (known risk):** DFL defines alive/dead timing and possession hand-over its own way, and that may differ from the organisers' GT (whistle vs line crossing; touch vs control). Boundary placement tuned on IDSSE inherits DFL's convention.
- The period seam joins rows a halftime apart. Ball kinematics, stillness and jump counts are now masked at every block edge (in the shared extractor). Anything temporal on IDSSE must split on `half`.
