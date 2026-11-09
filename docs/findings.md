# Findings from label-free checks on BAR-ATH

All of these come from the provided tracking and Smart Tagging only, with no ground truth and no video. They are allowed under the 2026-11-03 rules as automatic checks, and none of them is used to tune anything.

## 2026-11-09: feature extractor sanity (`extract_features.py --report`)
- **Parsing is correct.** Ball seen 59.3% and mean tracked 20.49 both match DATA.md.
- **Missing speeds:** 1.63 tracked slots per frame have a position but no speed. Speed features should use velocity computed from positions, not the raw speed field.
- **Nearest player** (when the ball position is known): home 56.4%, away 43.6%.
- **Ball outside the pitch:** 2.5% of graded frames.
- **Speeds:** ball median 5.1 m/s, p99 24.8; player median 1.5 m/s.
- **`ball_raw_f2` is undocumented.** Range 0–7.3, median 1.31. Don't use it until confirmed. (For comparison, IDSSE's ball z is in metres.)
- **The ball vanishes in long blocks, not flickers.** There are 150 missing runs; 132 of them are over 5 s and hold 99.7% of missing frames. Gap interpolation fills only 51 frames.

## 2026-11-09: what happens while the ball is missing (`diagnose_ball_gaps.py`)
132 long gaps cover 64,446 frames (40.6% of graded). Median length 11.6 s, longest 221.5 s.

| | gap | ball seen |
|---|---|---|
| share of players walking (median) | **0.95** | 0.60 |
| mean player speed (median) | **1.04 m/s** | 2.00 m/s |
| inside any set-piece tag | 21.7% | 31.7% |
| inside a `goals` tag (H + A) | **22.6%** | 2.1% |

- **69 of 132 gaps** have a set-piece tag starting inside them or within 1 s after.
- **Where the ball was last seen:** in a penalty area 26% of the time (2× the 13% base rate), outside the pitch 16%. Speed when last seen is low (median 2.2 m/s).
- **Where it came back:** outside the pitch 21%, in a box 29%, at the centre spot 5%. The ball is still over the next second in **only 2%** of cases, and it moved a median of 11 m while missing.

**Interpretation:** the long gaps are overwhelmingly stoppages. The tracker loses the ball while it is dead and picks it up again once it is played, so **reappearance ≈ the restart**. A minority of gaps (lower quartile of walking share, 0.86) may still be live play.

**Consequence:** build the dead-ball detector before the broadcast alignment.
