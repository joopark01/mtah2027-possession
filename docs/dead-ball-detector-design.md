# Dead-ball detector: design v1 (2026-11-09, draft for approval)

**Goal:** for every graded frame, decide alive or dead, and for each dead (D) segment name the team that restarts. The output feeds the possession decoder:
- a per-frame `p_dead`
- a list of D segments `[start, end)`, each with `restart_team` ∈ {H, A, unknown} and a confidence

**Why first:** on BAR-ATH, the long ball-missing gaps (40.6% of graded frames) behave like stoppages (see `findings.md`). Under spellF1, every missed or spurious D splits or merges possessions.

**Rules this design obeys:** every parameter is published, follows from the Laws of the Game or physics, or is tuned on **external** data (IDSSE). Nothing is tuned on BAR-ATH ground truth or video.

## 1. Evidence, per frame
| cue | signal (from `features_tactical.parquet`) | why |
|---|---|---|
| ball out of play | `ball_out_m` > margin | Law 9: out when the whole ball crosses the line. Margin = ball radius (0.11 m) + tracking error. |
| long ball gap | inside a ball-missing run ≥ `min_gap` | findings: these are mostly stoppages |
| players stop | `all_walking_frac` high, `all_mean_speed` low (from computed velocity, not the raw speed field) | the whistle → collective deceleration |
| placed ball | ball seen and `ball_still_frac` high, players walking | Laws 13–17: restarts are taken from a stationary ball |
| set piece announced | `st_set_pieces_*` active | Smart Tagging locates restarts (opens ~3.5 s early, so it locates but doesn't time) |
| goal | `st_goals_*` | the kickoff follows; long celebration gap |
| kickoff formation | each team in its own half, ball within 9.15 m of the centre spot | Law 8 |
| known dead | 110,808–113,741 (2nd half graded before its kickoff) | published |

## 2. Segments
1. **Candidates:** union of (a) long ball gaps where players slow, (b) ball-out spans, (c) placed-ball spans, and (d) the published pre-kickoff span.
2. **Live gaps:** a long ball gap where players keep running is not D. Those frames go to the team-shape model later.
3. **Merging:** join D candidates separated by a short alive blip (< `min_alive`). A spurious alive blip inside a stoppage costs extra spells.

## 3. Boundaries (where the points are)
- **End = restart.** The first frame the ball is seen *moving* after the window (findings: only 2% of reappearances are still). For placed balls, the first frame the ball leaves the still state. Tracker reacquisition lag is set to 0 a priori. It can't be measured externally, so it's flagged as a risk.
- **Start = play stops.** Use the first available of:
  - the frame the ball is seen crossing a line;
  - the onset of collective deceleration (a change point in team speed), shifted by the whistle→deceleration lag measured on IDSSE;
    *(v1: not a change point. The onset is the start of a threshold run: smoothed walking share and mean speed both past their thresholds, the run reaching back at most `dead_onset_max_lookback_frames` before the vanish frame. The lag is grid-tuned on IDSSE.)*
  - otherwise, the ball-vanish frame.

## 4. Who restarts
In priority order:
1. **Set-piece tag team** (H/A). `NA` means a throw-in with no team, so fall through.
2. **Goal tag:** the conceding team kicks off.
3. **Geometry:** ball out over a goal line → goal kick (defenders) or corner (attackers), decided by the attack direction and the last-seen side. Ball out over a touchline → the team of the nearest player at reappearance.
4. **Fallback:** the team of the nearest player when the ball reappears.

**v1 status (2026-11-15):** item 3's goal-line rule (goal kick vs corner) is deferred to **v2**, because it needs a last-touch proxy. v1 uses items 1, 2 and 4. For a touchline restart, item 3 gives the same answer as item 4.

## 5. Parameters (go into `config.json` with provenance)
| parameter | basis |
|---|---|
| `out_margin_m` | physics (ball radius + tracking error), checked on IDSSE |
| `min_gap_frames`, walking-share / speed thresholds, `min_alive` | tuned on IDSSE (degraded) |
| whistle→deceleration lag, change-point window | tuned on IDSSE |
| still-ball speed | already in config (Laws 13–17) |
| reacquisition lag | a_priori = 0 (cannot be measured externally) |
| `min_dead_frames` | tuned on IDSSE (degraded), 2026-11-16: 0 (off), all grid values tied |

## 6. Validation
- **IDSSE, held-out matches:** D-segment precision and recall at IoU ≥ 0.5, the distribution of start/end errors in frames, and restart-team accuracy. This needs the IDSSE per-frame ball status, which kloppy confirms exists (BallStatus / BallPossession).
- **BAR-ATH plausibility, no labels:**
  - D share of graded frames;
  - stoppage count per half (the 5 IDSSE tuning matches have a mean of 104.8 D segments per match, range 99–113, from the DFL BallStatus labels split per half);
  - restart-type mix vs Smart Tagging;
  - no D segment shorter than ~1 s (expectation only; v1 still predicts some, and `dead_min_dead_frames` tuned to 0, see `docs/findings.md`).

## 7. Build order
1. Download IDSSE (`features/external/fetch_idsse.py` → `external_data/idsse/`).
2. External adapter → the same feature table for IDSSE.
3. Rule-based v1 with the first-principles values; measure on IDSSE.
4. Tune the IDSSE-tuned parameters on 5 matches and validate on 2.
5. Run on BAR-ATH; plausibility checks; hand over to the decoder.

## Open questions
- Does the ground-truth D start at the whistle, the ball crossing the line, or later? The rules allow reading the docs and inspecting the data to understand the definition, but not fitting to it. CHALLENGE.md says D = dead per the corrected tracking, so IDSSE's ball-status convention is the best external proxy.
- Throw-in team: is "nearest player at reappearance" good enough, or is the side of the pitch plus the last touch better?
