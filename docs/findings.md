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

## 2026-11-15: ball-loss simulation (`features/external/degrade_ball.py`)
Imitates BAR-ATH's tracker on clean IDSSE ball tracking. Each DFL dead span maps to the gap [dead start + onset lag, restart + reacquisition lag), clipped to its period. Each lag is either a fixed int or, in the active config, an integer drawn per dead span uniform on [lo, hi]. If that gap is at least 126 frames, it is dropped with probability p; a shorter one is dropped with a lower probability p_short. On top come Poisson live-play gaps (log-uniform length) and 1–12-frame flickers. Each (match, seed) gets its own RNG, `default_rng([seed, match_index])`.

**Active simulator: a priori (decided 2026-11-15).** The user decided that the simulator must no longer be fitted to BAR-ATH aggregates. All 8 `ball_loss_*` values in `features/config.json` are now `a_priori`, with `tuned_on` null:

| key | value | reason |
|---|---|---|
| `dead_drop_prob` | 1.0 | User's decision: a tactical-camera tracker loses the ball whenever it is dead for ≥ 5 s (held, off the pitch, out of shot). |
| `short_gap_drop_prob` | 0.5 | No prior knowledge whether a brief stoppage (< 5 s) loses the ball, so the maximum-ignorance value. |
| `onset_lag_frames` | per span uniform on [0, 100] (0–4 s) | The tracker keeps following the ball for a moment after DFL marks it dead (rolling out, being collected). Was a fixed 50; changed 2026-11-15 (user's a priori choice) so the simulator no longer fixes the dead-ball edge exactly. |
| `reacquisition_lag_frames` | per span uniform on [0, 25] (0–1 s), clipped to the period end | Not identifiable from aggregates (sensitivity below). Was a fixed 0; changed for the same reason. |
| `live_gaps_per_hour`, `live_gap_frames_range` | 30, [126, 500] (5–20 s) | Occlusion in crowds and on aerial balls. Nonzero on purpose, so that "ball missing" on degraded IDSSE is not a proxy for "dead" and a detector has to use player cues. A round guess. |
| `flickers_per_hour`, `flicker_frames_range` | 10, [1, 12] | Brief detector dropouts. A round guess. |

These values were chosen without fitting, but they are **not blind**: the authors had already seen BAR-ATH's tracking aggregates (above) and the earlier fitted values. They were fixed before the tables were rebuilt and were not revised afterwards. A build with only a priori keys does not need `tune_log.json`. Each output's provenance records `"ball_loss_variant": "a_priori"` (or the `--ball-loss-params` file path) and the full `degrade_params`.

The lag spreads were fixed before the rebuild and not revised after. Even a span dropped with probability 1.0 now has random edges.

Rebuilt 2026-11-15 with `--all --seeds 0 1 2` (21 tables), randomised lags. Same layout as the fitted table below. The table is **descriptive only**: BAR-ATH is shown for comparison, nothing was changed in response, and the held-out numbers are reported separately.

| | BAR-ATH (comparison only) | tuning (5 × 3) | held out (2 × 3, evaluation only) |
|---|---|---|---|
| ball seen | 59.3% | 48.1% (39–55% per run) | 50.3% |
| runs ≥ 5 s | 132 | 157.7 | 146.0 |
| share of missing frames in runs ≥ 5 s | 99.7% | 99.4% | 99.4% |
| runs of 1–12 frames | ~16 | 16.8 | 16.8 |
| runs of 13–125 frames | ~2 | 5.6 | 4.9 |
| long-gap median | 11.6 s | 15.1 s | 15.4 s |
| longest | 221.5 s | 136 s | 187 s |
| bins 5-10 / 10-20 / 20-40 / 40-80 / 80-160 / 160+ s | 53 / 35 / 31 / 11 / 1 / 1 | 46.5 / 59.8 / 34.4 / 14.3 / 2.5 / 0.2 | 43.0 / 53.1 / 33.8 / 13.4 / 1.6 / 1.1 |
| ball still 1 s after a long gap ends | 2% | 1.8% | 2.1% |

(The fixed-lag a priori build it replaces, onset 50 and reacquisition 0: tuning seen 48.6%, 157.9 long runs, median 14.7 s; held out 52.4%, 138.2, 15.4 s.)

**Edge check (tuning matches only, 5 × 3; descriptive, nothing changed in response).** Over the long (≥ 126-frame) dropped runs that overlap DFL dead spans: vanish offset = run start − first overlapping DFL dead onset; reappear offset = run stop − DFL restart of the last overlapped dead span the run has passed (runs ending at the period end excluded).

| | n | min | q25 | median | q75 | max | distinct values |
|---|---|---|---|---|---|---|---|
| ball vanish − DFL dead onset (frames) | 1376 | 0 | 26 | 50 | 76 | 100 | 101 |
| ball reappear − DFL restart (frames) | 1365 | 0 | 6 | 12 | 19 | 25 | 26 |

Both match their uniform spreads. Of 2,113 long runs, 737 are live-play gaps (no dead overlap). 7 span more than one dead span, and 6 end inside a later dead span that was not dropped (the reacquisition lag reached past a live interval shorter than itself).

On degraded IDSSE the edges of a ball-loss D spell are no longer at a fixed offset from DFL's, but their spreads (0–4 s onset, 0–1 s reacquisition) are themselves assumptions. A detector's ball vanish/reappear offsets stay `a_priori` and are never tuned on degraded tables.

### Fitted variant (unused, pending organisers' confirmation)
These are the earlier values, fitted to BAR-ATH's **label-free tracking aggregates** (above and DATA.md, never GT). They are kept with their full provenance (`tuned_external`, `tuned_on`, `target`, and the tune-log reference) in `features/external/ball_loss_fitted_bar_ath.json`. `external_data/idsse_degraded/tune_log.json` stays as their audit trail. To switch back, if the organisers confirm this is allowed:

    python features/external/degrade_ball.py --all --seeds 0 1 2 \
        --ball-loss-params features/external/ball_loss_fitted_bar_ath.json

`--tune` and `--sensitivity` remain available. They write only the tune log or stdout, never `config.json`.

**Tuning.** A full joint grid of 14,400 combinations ran on the 5 tuning matches × seeds 0, 1; the held-out matches were never loaded. Every row is in `external_data/idsse_degraded/tune_log.json` (2026-11-15T14:57). The config was frozen before the held-out run. Chosen: p 0.95, p_short 0.2, onset 100 frames (4 s), reacquisition 0 (a priori), 20 live gaps/h of 126–250 frames, 9 flickers/h. Every value is interior to its grid. The surface is flat near the optimum (the top 10 are within 0.45 of the best loss of 13.44), so the values are only weakly identified.

Seeds 0–2, built 2026-11-15 before the switch. Counts are scaled to BAR-ATH's 158,857 graded frames. "Longest" is the mean over runs of each match's longest gap.

| | target (BAR-ATH) | tuning (5 × 3) | held out (2 × 3, evaluation only) |
|---|---|---|---|
| ball seen | 59.3% | 59.5% (53–67% per run) | 61.5% |
| runs ≥ 5 s | 132 | 127.2 | 113.1 |
| share of missing frames in runs ≥ 5 s | 99.7% | 99.5% | 99.5% |
| runs of 1–12 frames | ~16 | 16.3 | 16.1 |
| runs of 13–125 frames | ~2 | 2.6 | 2.5 |
| long-gap median | 11.6 s | 13.1 s | 13.9 s |
| longest | 221.5 s | 132 s | 184 s |
| bins 5-10 / 10-20 / 20-40 / 40-80 / 80-160 / 160+ s | 53 / 35 / 31 / 11 / 1 / 1 | 53.3 / 32.1 / 27.8 / 11.9 / 1.9 / 0.2 | 45.5 / 23.8 / 29.5 / 11.6 / 1.6 / 1.1 |
| ball still 1 s after a long gap ends | 2% | 2.2% | 1.8% |

**Sensitivity** (`--sensitivity`, tuning matches × seeds 0, 1, other parameters at the fitted values; earlier build, before the a priori switch):

| reacq (frames) | onset (frames) | seen | runs ≥ 5 s | 13–125 | median (s) | bins 5/10/20/40/80/160+ | still after |
|---|---|---|---|---|---|---|---|
| 0 | 50 | 56.0% | 134.5 | 1.9 | 14.0 | 56/30/32/14/2/0 | 2.0% |
| 0 | 100 | 59.5% | 127.3 | 2.9 | 13.1 | 53/32/28/12/2/0 | 2.1% |
| 0 | 150 | 62.9% | 118.9 | 3.4 | 12.6 | 52/28/26/10/2/0 | 2.1% |
| 12 | 50 | 55.2% | 136.0 | 1.3 | 14.2 | 55/31/33/14/2/0 | 1.7% |
| 12 | 100 | 58.7% | 128.9 | 2.7 | 13.3 | 53/32/29/12/2/0 | 1.7% |
| 12 | 150 | 62.1% | 120.5 | 3.3 | 12.7 | 52/29/26/11/2/0 | 1.9% |
| 25 | 50 | 54.3% | 136.5 | 1.8 | 14.6 | 54/29/36/15/2/0 | 1.8% |
| 25 | 100 | 57.8% | 131.3 | 1.9 | 13.4 | 54/33/29/12/2/0 | 1.8% |
| 25 | 150 | 61.2% | 123.0 | 3.4 | 12.9 | 53/30/26/12/2/0 | 2.1% |

The onset lag mostly moves the seen share (about 3.5 pp per 2 s). The reacquisition lag moves almost nothing: 1 s of lag costs about 1.7 pp of seen share and leaves the "still after" share flat. **The aggregates cannot identify the reacquisition lag.** With this variant's fixed lags, both edges of every D spell that loses the ball on degraded IDSSE are fixed by simulator assumptions: the start by the tuned onset lag, the end by the a-priori reacquisition lag of 0. Boundary-placement results on degraded IDSSE inherit both.

**What still didn't fit:**
- Too few very long gaps (160 s+: 0.2 vs 1), and the median runs high (13.1 vs 11.6 s).
- The held-out pair has fewer long runs (113 vs 132) and a mid-range dip (10–20 s: 24 vs 35). This is reported only: no simulator change will be made in response to held-out numbers. With this variant, `--all` refuses to run unless the active values equal the tune log's best row, and each output records the log's sha256 and timestamp.
- "Ball still 1 s after reappearing" is computed slightly differently for BAR-ATH (`diagnose_ball_gaps.py`: clipped at the array end, NaN counted as not still) and for the simulator (within the period, NaN skipped). The 2% vs 2.2% comparison is approximate. The metric is not in the loss.
- The seen share varies by ±7 pp between matches and seeds, because each IDSSE match has its own dead share.
- The BAR-ATH target includes 2 long runs (540 frames, 21.6 s, all inside the span) in the 2nd-half pre-kickoff block [110808, 113742), which IDSSE cannot produce because its periods start at kickoff. That is 0.34% of graded frames and about 1.5% of the long-run count: a known small mismatch, and TARGET is kept unchanged. It does not explain the 160 s+ shortfall.

## 2026-11-15: tuning harness and trivial baseline (`features/external/harness.py`)

Scoring is `evaluation/possession_metrics.py` (`pm.evaluate`). 26 X rows (one more than the widest transition tolerance) are inserted at the period seam in GT and prediction, so no spell or transition match crosses halftime. GT is the DFL proxy label. The predictor never sees `label` or `ball_dropped`. Degraded tables use the a priori ball-loss variant, 3 seeds each; the tune table below was re-run on the randomised-lag tables (2026-11-15). Clean tables get one run per match. `baseline_nearest`: when the ball position is known, predict the nearest player's team; otherwise D.

**Tune split (J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ), mean (min–max) over runs:**

| tables | runs | spell_f1_mean | spell_f1@0.5 | macro_f1 | alive_dead_f1 | team_acc_gt_alive | n_gt_spells | n_pred_spells | fragmentation |
|---|---|---|---|---|---|---|---|---|---|
| degraded | 15 | 0.106 (0.08–0.13) | 0.092 (0.07–0.11) | 0.720 (0.69–0.75) | 0.861 (0.81–0.90) | 0.617 (0.55–0.67) | 454 (401–511) | 1765 (1529–1994) | 3.45 (2.78–4.25) |
| clean | 5 | 0.100 (0.08–0.12) | 0.086 (0.07–0.10) | 0.359 (0.33–0.38) | 0.704 (0.65–0.74) | 0.764 (0.75–0.78) | 454 (401–511) | 2092 (1981–2240) | 3.77 (3.32–4.43) |

**HELD OUT (J03WPY, J03WR9). Evaluation only: nothing was tuned or selected on these numbers.** Both looks are logged in `external_data/harness/holdout_log.json`. The degraded row was made on the earlier **fixed-lag** tables (onset 50, reacquisition 0) and was not re-run on the randomised-lag tables.

| tables | runs | spell_f1_mean | spell_f1@0.5 | macro_f1 | alive_dead_f1 | team_acc_gt_alive | n_gt_spells | n_pred_spells | fragmentation |
|---|---|---|---|---|---|---|---|---|---|
| degraded | 6 | 0.103 (0.10–0.11) | 0.090 (0.09–0.09) | 0.757 (0.74–0.77) | 0.896 (0.89–0.91) | 0.681 (0.66–0.70) | 396 (370–421) | 1792 (1685–1886) | 3.93 (3.87–4.02) |
| clean | 2 | 0.095 (0.09–0.10) | 0.083 (0.08–0.09) | 0.376 (0.37–0.39) | 0.725 (0.72–0.74) | 0.797 (0.79–0.81) | 396 (370–421) | 1991 (1864–2117) | 4.06 (4.03–4.09) |

**Failure modes, read from the aggregates:**
- **Fragmentation is the main failure.** The baseline predicts 3.9x (degraded, tune) to 4.6x (clean) as many spells as the GT (n_pred_spells / n_gt_spells). Nearest-player identity flips during contests and passes, so spell_f1 stays near 0.1 even where the frame metrics look fine.
- **The degraded tables score better than the clean ones on frames.** On clean tables the ball is always known, so the baseline never predicts D. That gives macro_f1 0.36 and alive_dead_f1 0.70, and the gap is all false alive during dead spans. On degraded tables a missing ball mostly means a dead span, so "missing → D" gets alive/dead right much of the time (alive_dead_f1 0.86). That is a property of the simulator, which hides the ball mostly in dead spans, and it says nothing about competence. The ball-loss edges are now randomised a priori per dead span (onset 0–4 s, reacquisition 0–1 s), so they no longer sit at a fixed offset from the DFL edges. A detector's ball vanish/reappear offsets stay `a_priori` and are never tuned on degraded tables. Residual risk: the spreads are themselves assumptions, so degraded alive/dead scores remain only as realistic as those spreads.
- **Degraded vs clean spell_f1 (0.106 vs 0.100) is within noise** (SPELL_NOISE_BAND 0.04).
- **The team call is the second error.** Team accuracy is 0.76 on clean tables, where the baseline predicts alive almost everywhere. On degraded tables it drops to 0.62, because alive frames with a hidden ball are predicted D and the strict metric counts them as wrong.

## 2026-11-15: dead-ball detector v1, untuned (`features/dead_ball.py`)

First-principles values from `features/config.json` (`dead_*`), including the onset look-back bound `dead_onset_max_lookback_frames` = 125 (5 s, a priori). The main session added the bound before any tuning, because of a design concern (an unbounded backward search), not because of a score. Nothing was tuned, and no value was changed after these scores were seen. Tune split only; the held-out pair was not run. Scored with `harness.py --predictor dead_ball_v1` (alive frames: nearest team, carried forward over unknown-ball frames).

**Tune split (J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ), mean (min–max) over runs, vs `baseline_nearest` (above):**

| predictor | tables | runs | spell_f1_mean | spell_f1@0.5 | macro_f1 | alive_dead_f1 | team_acc_gt_alive | n_pred_spells | fragmentation |
|---|---|---|---|---|---|---|---|---|---|
| baseline_nearest | degraded | 15 | 0.106 (0.08–0.13) | 0.092 (0.07–0.11) | 0.720 (0.69–0.75) | 0.861 (0.81–0.90) | 0.617 (0.55–0.67) | 1765 (1529–1994) | 3.45 (2.78–4.25) |
| dead_ball_v1 | degraded | 15 | 0.120 (0.09–0.14) | 0.106 (0.08–0.13) | 0.756 (0.73–0.80) | 0.917 (0.90–0.95) | 0.730 (0.70–0.75) | 1704 (1471–1953) | 3.26 (2.60–4.11) |
| baseline_nearest | clean | 5 | 0.100 (0.08–0.12) | 0.086 (0.07–0.10) | 0.359 (0.33–0.38) | 0.704 (0.65–0.74) | 0.764 (0.75–0.78) | 2092 (1981–2240) | 3.77 (3.32–4.43) |
| dead_ball_v1 | clean | 5 | 0.115 (0.09–0.14) | 0.101 (0.08–0.12) | 0.776 (0.76–0.81) | 0.933 (0.92–0.96) | 0.757 (0.74–0.77) | 2124 (2014–2271) | 4.09 (3.55–4.84) |

n_gt_spells is 454 (401–511) throughout.

**D segments vs GT (DFL) D runs, per match, split at the period seam:** degraded 103.6 predicted vs 104.8 GT (95–109 vs 99–113 per run); clean 141.2 vs 104.8 (122–150). Predicted D share: degraded 32.6–47.2% vs GT 40.7–51.7%; clean 31.3–46.1%. One `detect` call takes about 0.013 s per match table.

- **Alive/dead works on clean tables too** (alive_dead_f1 0.93, from 0.70), so the player cues carry it, not just "ball missing". On clean tables the ball is never missing, so every D there comes from the ball-out, placed-ball and stopped-player cues.
- **spell_f1 barely moves** (0.106 → 0.120 degraded, 0.100 → 0.115 clean; the difference is inside SPELL_NOISE_BAND 0.04). Fragmentation is still 3–4x: the nearest-player team call during live play is the failure, which is the decoder's job.
- **Under-calls D** (predicted share 4–9 pp below GT): consistent with a late start (the stopped onset minus 1 s) or short stoppages that never stop the players. The clean tables over-count D segments (141 vs 105), so some stoppages are split or have spurious short D in live play. Not investigated further. Tuning comes next, through the harness on the tune split.
- **The look-back bound barely matters here.** Without it (the first v1 run): degraded spell_f1 0.120, alive_dead_f1 0.917, 103.3 D segments. On clean tables nothing changes, because there are no ball gaps.

## 2026-11-16: dead-ball detector tuning (`features/external/tune_dead_ball.py`)

Data: IDSSE (Bassek, Rein, Weber et al., Sci Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y, CC BY 4.0). We tuned only the six player-based keys in `dead_ball.TUNABLE`. The ball vanish/reappear offsets, the out margin, the placed-ball threshold and the look-back bound kept their a priori values. The tuning data were the degraded (a priori ball-loss) tables of the 5 tuning matches × seeds 0–2, 15 runs per combo, loaded only through `harness.run(split="tune")`. The held-out pair was never loaded. The objective was the mean `spell_f1_mean` (harness → `pm.evaluate`). The tie rule was declared in the log before the first score: among combos within 0.002 of the best, take the highest mean `alive_dead_f1`; if still tied, take the combo closest to the a priori values in grid-index distance. The search was a full joint grid, not coordinate descent.

**Run 1** (`external_data/dead_ball_tune/tune_log_run1.json`, 2026-11-15T23:58, 10,000 combos). Grid: lag {0, 12, 25, 50, 75}, smoothing {6, 12, 18, 25}, walking {0.6, 0.7, 0.8, 0.9, 0.95}, speed {1.0, 1.25, 1.5, 2.0, 2.5} m/s, min_gap {25, 50, 100, 150}, min_alive {0, 12, 25, 50, 75}. Chosen: lag 12, smoothing 6, walking 0.7, speed 2.0, min_gap 50, min_alive 0. Smoothing (6) and min_alive (0) were on grid edges.

**Run 2, the edge extension** (`external_data/dead_ball_tune/tune_log.json`, 2026-11-16T01:29, 15,000 combos, sha256 8df0242a…). The smoothing grid was extended once, down to {0, 3}. min_alive could not be extended, because 0 (no merging) is its hard lower bound. Run 2 chose lag 0 (a lower edge, but a physical bound: a negative lag would start D after the deceleration), smoothing 0 (again an edge and a physical bound: no smoothing), walking 0.7, speed 2.0 m/s, min_gap 50 and min_alive 25. **Two keys are not interior: lag and smoothing, both at their physical minimum.** The single extension was used, so this is reported, not extended again. These are the values in `config.json`. `harness.dead_ball_v1` and `tune_dead_ball.py --check` refuse to run if the config differs from the log's chosen row.

| key | a priori | chosen |
|---|---|---|
| dead_decel_lag_frames | 25 | 0 (edge) |
| dead_smooth_half_window_frames | 12 | 0 (edge) |
| dead_walking_frac_min | 0.8 | 0.7 |
| dead_mean_speed_max_m_s | 1.5 | 2.0 |
| dead_min_gap_frames | 50 | 50 |
| dead_min_alive_frames | 25 | 25 |

**Flatness: spell_f1 does not identify these keys.** Across all 15,000 combos, mean spell_f1 spans only 0.111–0.122, a quarter of SPELL_NOISE_BAND (0.04). The top 10 combos lie within 0.00015 (run 1: 0.00009), and 1,768 combos are within the 0.002 tie band. In practice the choice was made by `alive_dead_f1` inside that band. The spell_f1 optimum itself is lag 0, smoothing 3, walking 0.8, speed 2.0, min_gap 150, min_alive 75 (0.1225, alive_dead_f1 0.923). The alive_dead_f1-optimal combo is lag 0, smoothing 0, walking 0.6, speed 2.0, min_gap 50, min_alive 12 (alive_dead_f1 0.9505, spell_f1 0.1190). The chosen combo scores alive_dead_f1 0.9494 and spell_f1 0.1212. Only walking and speed move alive_dead_f1 by a lot (0.95 → 0.77 and 0.83 at the strict ends).

**Tune split after tuning (J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ), mean (min–max) over runs.** The held-out pair was not run.

| predictor | tables | runs | spell_f1_mean | spell_f1@0.5 | macro_f1 | alive_dead_f1 | team_acc_gt_alive | n_pred_spells | fragmentation |
|---|---|---|---|---|---|---|---|---|---|
| baseline_nearest | degraded | 15 | 0.106 (0.08–0.13) | 0.092 (0.07–0.11) | 0.720 (0.69–0.75) | 0.861 (0.81–0.90) | 0.617 (0.55–0.67) | 1765 (1529–1994) | 3.45 (2.78–4.25) |
| dead_ball_v1 untuned | degraded | 15 | 0.120 (0.09–0.14) | 0.106 (0.08–0.13) | 0.756 (0.73–0.80) | 0.917 (0.90–0.95) | 0.730 (0.70–0.75) | 1704 (1471–1953) | 3.26 (2.60–4.11) |
| dead_ball_v1 tuned | degraded | 15 | 0.121 (0.09–0.14) | 0.107 (0.08–0.13) | 0.787 (0.77–0.81) | 0.949 (0.93–0.97) | 0.722 (0.69–0.75) | 1715 (1486–1974) | 3.31 (2.65–4.20) |
| baseline_nearest | clean | 5 | 0.100 (0.08–0.12) | 0.086 (0.07–0.10) | 0.359 (0.33–0.38) | 0.704 (0.65–0.74) | 0.764 (0.75–0.78) | 2092 (1981–2240) | 3.77 (3.32–4.43) |
| dead_ball_v1 untuned | clean | 5 | 0.115 (0.09–0.14) | 0.101 (0.08–0.12) | 0.776 (0.76–0.81) | 0.933 (0.92–0.96) | 0.757 (0.74–0.77) | 2124 (2014–2271) | 4.09 (3.55–4.84) |
| dead_ball_v1 tuned | clean | 5 | 0.117 (0.09–0.14) | 0.103 (0.08–0.13) | 0.804 (0.78–0.83) | 0.961 (0.95–0.98) | 0.754 (0.74–0.77) | 2108 (1988–2264) | 4.06 (3.52–4.85) |

These are in-sample numbers: the degraded tuned row is the tuning objective itself. Tuning raised alive_dead_f1 by 0.03 and left spell_f1 unchanged within noise (+0.001). Spell F1 is still limited by fragmentation in live play, which is the decoder's job.

**Clean cross-check** (`external_data/dead_ball_tune/crosscheck.json`). Each key was varied over its grid with the others at the chosen values, on the clean tuning tables (5 runs), next to the degraded rows from the log. The chosen values were not changed in response. Each cell is spell_f1 / alive_dead_f1:

| key | values | degraded | clean |
|---|---|---|---|
| lag | 0* 12 25 50 75 | .1212/.949, .1205/.949, .1193/.948, .1169/.943, .1147/.937 | .117/.961 for every value (clean tables have no ball gaps) |
| smoothing | 0* 3 6 12 18 25 | .1212/.949, .1211/.949, .1214/.949, .1211/.948, .1212/.948, .1210/.947 | .1170/.961, .1172/.961, .1173/.960, .1172/.960, .1172/.959, .1171/.959 |
| walking | 0.6 0.7* 0.8 0.9 0.95 | .119/.951, .121/.949, .122/.929, .119/.861, .116/.768 | .118/.973, .117/.961, .116/.944, .111/.895, .107/.841 |
| speed | 1.0 1.25 1.5 2.0* 2.5 | .118/.826, .121/.894, .122/.937, .121/.949, .121/.949 | .110/.870, .113/.917, .116/.948, .117/.961, .117/.961 |
| min_gap | 25 50* 100 150 | .121/.949, .121/.949, .121/.949, .121/.948 | .117/.961 for every value |
| min_alive | 0 12 25* 50 75 | .120/.949, .121/.949, .121/.949, .121/.949, .121/.948 | .114/.959, .116/.960, .117/.961, .117/.963, .117/.963 |

\* = chosen. **Flags** (rule: the clean optimum is more than one grid step from the degraded choice, or the clean score at the choice is more than 0.01 below the clean optimum):
- **dead_smooth_half_window_frames**: the clean spell_f1 optimum is 6, against the choice of 0 (2 steps). The difference is 0.0003, far inside the noise band.
- **dead_min_alive_frames**: the clean spell_f1 optimum is 75, against 25 (2 steps). The difference is 0.0002.
- **dead_walking_frac_min**: clean alive_dead_f1 at the choice (0.7) is 0.961, which is 0.012 below the clean optimum of 0.973 at 0.6 (the grid edge). On clean tables, a looser walking threshold keeps helping alive/dead. On degraded tables the profile is flatter (0.951 vs 0.949). This is the one flag of practical size.

**Transfer risk.** The degraded tables degrade only the ball; the player tracking is clean IDSSE. Smoothing 0, lag 0 and walking 0.7 were chosen exactly where clean IDSSE player tracking and the noisier BAR-ATH tactical tracking would differ most. The a priori reason for smoothing was single-frame tracking noise, which IDSSE does not have, so tuning could not see what removing it costs. The choice is also unstable inside the flat band: from run 1 to run 2, lag went 12 → 0 and min_alive 0 → 25, although only the smoothing grid was extended. The script hash differs between the two logs (`script_sha256` 1437872b… vs 3bae91fb…) because `tune_dead_ball.py` gained `--extend` and `--previous` between them. The run 2 log's header has two extra fields, `extended` and `previous_run`, and the harness and config hashes are identical in both logs. Treat these three values as weakly supported on BAR-ATH.

## 2026-11-16: dead-ball detector held-out evaluation and BAR-ATH plausibility

**Held-out pair (J03WPY, J03WR9): first look**, `harness.py --predictor dead_ball_v1 --holdout` with the frozen tuned values. Logged in `external_data/harness/holdout_log.json` (entries at 2026-11-16T03:06:19, one each for degraded and clean tables):

| tables | spell_f1 | macro | alive_dead_f1 | team | n_pred_spells |
|---|---|---|---|---|---|
| degraded | 0.115 | 0.797 | 0.950 | 0.746 | 1681 |
| clean | 0.113 | 0.820 | 0.965 | 0.784 | n/a |

- alive_dead_f1 matches in-sample (0.949 degraded): the alive/dead split transfers across matches.
- spell_f1 sits inside the flat tuning band (0.111–0.122), so it says nothing either way about overfitting.

**Second look: D-segment counts** (`features/external/dead_segment_counts.py`). Logged by hand as predictor `dead_ball_v1_dsegment_counts` (2026-11-16T03:06:59). The entry's `script_sha256` is harness.py's, not this script's, and its summary values are rounded. The script has since been restricted to the tuning matches, so it cannot repeat the held-out look. Nothing was changed in response to it. After this look the held-out pair is **no longer clean for judging any v2 change motivated by these counts**; a later held-out number would be reported as a further look.

| split | tables | pred D segs | true D segs | pred < 1 s | pred D share | true D share |
|---|---|---|---|---|---|---|
| tune | degraded | 119.5 | 104.8 | 10.1% | 44.5% | 45.6% |
| tune | clean | 143.6 | 104.8 | 13.2% | 42.4% | 45.6% |
| holdout | degraded | 126.5 | 94.5 | 16.8% | 42.6% | 43.1% |
| holdout | clean | 147.5 | 94.5 | 18.8% | 40.6% | 43.1% |

**Proposed v2: a minimum D length of 1 s, chosen a priori** (design doc section 6: no stoppage is shorter than about 1 s). The support comes from the tuning matches alone: 14–37% more predicted D segments than true ones, and 10.1–13.2% of predicted segments under 1 s, while the true labels have almost none. A spurious short D segment splits a possession spell. Removing it can instead merge two spells; that effect is unmeasured.

**Decision (user, 2026-11-16):** `dead_min_dead_frames` tuned on the tuning matches by spell_f1 over {0, 12, 25, 38, 50} frames (`tune_dead_ball.py --tune-one`, log `external_data/dead_ball_tune/tune_log_min_dead.json`). All five lay within the 0.001 tie band (25: 0.1222, 0: 0.1212), so the tie rule took **0 (off)**: removing short D segments removes about 16 of ~1,700 predicted spells per match and does not fix over-segmentation, which comes from the live-play team call. The `100-120 restarts` reference is replaced by the true IDSSE tuning-match mean, 104.8 D segments per match (range 99-113).

**BAR-ATH plausibility** (`features/dead_ball_report.py`; label-free aggregates, no ground truth, no video):

```
graded frames 158857, predicted D share 43.1%
  excluding the 2nd-half pre-kickoff span: D share 42.0%, 144 D segments
  half 1: 65 D segments
  half 2: 80 D segments
  total 145 D segments (design doc: a match has roughly 100-120 restarts)
  length s: shortest 0.08, quartiles 2.7 / 9.8 / 21.6, longest 221.2
  shorter than 1 s: 13 (9.0%)
  restart team: A 76, H 65, unknown 4
  D segments overlapping any set-piece tag, any team incl. N/A (or within 1 s after): 78 / 145
  D frames with the ball seen: 16.8%;  ball-missing graded frames called D: 88.1%
```

- **D share:** compare predicted with predicted. Without the pre-kickoff span (IDSSE periods start at kickoff), BAR-ATH is 42.0%, inside the IDSSE predicted split means (40.6–44.5%).
- **Segments:** 144 is above the IDSSE true mean of 104.8 per tuning match (range 99–113), as on IDSSE. Dropping the 13 under 1 s would leave about 131, still above the range.
- **Set-piece overlap:** 78 / 145 is an upper bound on tag-based restart calls, because the window runs 1 s past the restart. Smart Tagging coverage of throw-ins and goal kicks is unknown, so this is weak evidence either way.
- **Ball-missing frames:** 11.9% are called alive. No independent estimate of the live share of missing frames exists to compare it with.
