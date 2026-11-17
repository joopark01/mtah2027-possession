# MTAH 2027: possession per frame on BAR-ATH (technical document, first draft)

Draft of 2026-11-16. Sections marked **TODO** describe work that is not finished yet. Every number below comes from `docs/findings.md`, `features/config.json`, a tracked tune log in `results/tuning/`, or the held-out log `external_data/harness/holdout_log.json` (untracked; `external_data/` is gitignored). Each number says whether it is in-sample or held-out. No number here was computed on BAR-ATH ground truth or checked against the video.

## 1. Summary

**Task.** Label every frame of the BAR-ATH tactical video (199,319 frames, 158,857 graded) as `H` (home in possession), `A` (away in possession), `D` (dead ball) or `X` (ungraded). The ranking metric is spellF1: each possession, a maximal run of `H` or `A`, is matched one-to-one against the truth by temporal IoU, and the F1 is averaged over IoU thresholds 0.3–0.7.

**Approach.** The pipeline has no learned model and uses only the provided tactical tracking (ATD) and Smart Tagging. (1) A feature extractor turns the raw ATD into one row of ball, team and cross-team features per video frame. (2) A rule-based dead-ball detector marks the stoppages. On BAR-ATH the tracker loses the ball in long blocks, mostly while it is dead, so a long ball gap together with players walking is the main dead-ball cue. Ball-out-of-play and placed-ball cues are added, and Smart Tagging names the team that restarts. (3) A possession decoder runs a three-state Viterbi (home, away, contested/loose) over each alive span. It scores each frame by the nearest-player margin and charges a penalty for each change of state. When the ball position is unknown, a team-shape model scores the frame instead: a logistic regression on home-minus-away player-shape differences, fitted on IDSSE with the ball hidden. Contested frames take the preceding team's label, so a loose ball never splits a spell. Every parameter is published, follows from the Laws of the Game or physics, was chosen a priori, or was tuned on an external dataset (IDSSE, the 7 Bundesliga matches of Bassek et al. 2025). The IDSSE tuning tables were first degraded with a simulated BAR-ATH-style ball loss.

**Key numbers. All are in-sample, on the five IDSSE tuning matches with degraded ball tracking, 15 runs (5 matches × 3 seeds), scored with the organisers' `pm.evaluate` against DFL proxy labels.**

| pipeline | spell_f1_mean (in-sample) |
|---|---|
| nearest-player baseline | 0.106 |
| dead-ball detector + nearest team (tuned) | 0.121 |
| + possession decoder, tuning run 1 | 0.381 |
| + possession decoder, tuning run 2 | 0.391 |
| + team-shape score on ball-unknown frames, tuning run 3 (current config) | 0.418 |

**Held-out status.** The integrated pipeline (detector + decoder + team-shape model) has **not** been evaluated on the held-out IDSSE matches. That will happen once, as the final step, and it will be logged (section 2.4). The dead-ball detector on its own has already had two held-out looks; alive_dead_f1 on the held-out degraded tables was 0.950 (section 6.4).

## 2. Data and rules compliance

### 2.1 Provided data
- The BAR-ATH tracking, Smart Tagging and video are used only on the local machine. The derived feature table and the submission go to `data/derived/`, which is gitignored and never committed or uploaded.
- An organisers' clarification of 2026-11-03 forbids uploading the provided data to AI services. AI coding assistants were used for this entry. They never read rows of anything under `data/`: scripts ran locally and the assistant saw only their aggregate summaries. `.claude/settings.json` blocks direct reads of `data/`.
- Assets used: the tactical ATD (player and ball positions) and Smart Tagging. Only the `SET PIECES` and `GOALS` tags are used, and only to name the restart team. **Not used:** the broadcast video (alignment is pending, section 8), the homography, and the third raw ball field `ball_raw_f2` (undocumented on BAR-ATH, so it is kept unused).

### 2.2 No fitting on BAR-ATH
- Nothing was trained, fitted or tuned on the BAR-ATH ground truth or the video. That includes the public 5-minute ground-truth window: `docs/findings.md` records no score on it.
- There are no hand labels on BAR-ATH, and the output was never checked against the video.
- The BAR-ATH checks we did run are **label-free aggregates** of the tracking and Smart Tagging (D share, number of D segments, gap statistics). No value was ever fitted to them. They did, however, inform some design and a priori choices, as listed below, so the pipeline is not blind to them.
- **"Not blind" disclosure.** BAR-ATH label-free aggregates had been seen before these choices were made:
  - **Ball-loss simulator.** An earlier version was fitted to BAR-ATH's label-free tracking aggregates (never to GT). On 2026-11-15 the user decided not to use it until the organisers confirm such fitting is allowed. The fitted values are archived, unused, in `features/external/ball_loss_fitted_bar_ath.json`. The active values are a priori (section 3.2), but they were chosen by people who had already seen those aggregates and the fitted values.
  - **Dead-ball thresholds.** The a priori points of `dead_walking_frac_min` (0.8) and `dead_mean_speed_max_m_s` (1.5 m/s) were set after the BAR-ATH gap medians had been seen (walking share 0.95 and mean speed 1.00 m/s in long gaps, against 0.62 and 1.93 m/s with the ball seen). Their provenance says "not blind". The joint grid's tie rule falls back on grid-index distance to the a priori point. So these aggregates could influence the tuned choice indirectly, even though that fallback applies only to exact ties in alive_dead_f1. findings records that the choice inside the tie band was in practice made by alive_dead_f1 (section 6.1).
  - **`dead_min_dead_frames`.** This key was proposed after the label-free BAR-ATH report (13 of 145 D segments under 1 s) and after the held-out D-segment look (section 2.4). It was tuned on the tuning matches only and came out at 0 (off).

### 2.3 Parameter provenance
Every config key has a `_provenance` entry with `basis` ∈ {`published`, `law_of_game`, `physics`, `a_priori`, `tuned_external`}, plus `source`, `reason` and `tuned_on`. `extract_features.check_provenance()` refuses to run if any of the following holds:
- a key has no entry, or an entry has no key;
- a `tuned_external` key names no dataset or no target;
- a `tuned_external` key names BAR-ATH;
- a `tuned_external` key names a held-out IDSSE match as tuning data;
- a `tune_log` lies outside `results/tuning/`, is missing, or has a sha256 different from the recorded one.

Section 4 lists every key.

### 2.4 IDSSE split and held-out policy
- **Split, fixed 2026-11-09, before any results:** tune on J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ. Hold out J03WPY and J03WR9, for evaluation only. The split is hard-coded (`ef.IDSSE_TUNE`, `ef.IDSSE_HOLDOUT`). `harness.load_table` refuses to load a held-out table unless `allow_holdout=True` (CLI `--holdout`), and every held-out run is appended to `external_data/harness/holdout_log.json`.
- **Held-out policy (user decision, 2026-11-16):** no more per-component held-out looks. The full integrated pipeline gets one final held-out evaluation, it is logged, and nothing is changed after it.
- **Looks already taken (stated honestly).** The held-out pair is not untouched. `holdout_log.json` holds:
  - `baseline_nearest`, degraded and clean tables (2026-11-15T20:40:30);
  - `dead_ball_v1`, degraded and clean tables (2026-11-16T03:06:19). This is the detector's first look.
  - `dead_ball_v1_dsegment_counts` (2026-11-16T03:06:59), the detector's second look: D-segment counts. It was logged by hand, and its `script_sha256` is the harness's, not the script's.

  Nothing was tuned or selected on these numbers. Label-derived descriptive numbers on J03WPY and J03WR9 were also seen, but not used for selection:
  - the DFL D share (41–52%) and label-run counts (462–614), reported over all 7 matches in the adapter check;
  - the held-out columns of the degraded-table ball-loss aggregates (ball seen share, gap counts and lengths) for both simulator variants.

  After the second look the held-out pair is no longer clean for judging any change motivated by the D-segment counts. One later key, `dead_min_dead_frames`, was proposed after that look. It was tuned on the tuning matches only, and it came out at 0 (off), which leaves the detector identical to the version scored on held-out (section 5.3).

## 3. Method

The code is in `features/`. Every stage works per `half` block: nothing (smoothing, runs, merging, decoding) crosses a half edge or, on IDSSE, the period seam where the rows of the two periods sit next to each other. All intervals are half-open.

### 3.1 Feature extraction (`features/extract_features.py`)
- **Parsing.** The raw ATD is streamed into arrays [frames × slots × (x, y, speed)] plus the ball fields, and any unexpected format is an error. Row i is video frame i. Normalised coordinates are scaled by the published pitch size (105.3 × 68.0 m). The parse is verified against DATA.md: the ball is seen on 59.3% of frames and the mean number of tracked players is 20.49, both matching (findings, 2026-11-09).
- **Orientation.** Each team's attack direction per half is read off the kickoff formation (Law 8). The team centroids are averaged over a 1 s window next to the kickoff and must be at least 5 m apart along x, or the script stops.
- **Ball.** Gaps of up to 12 frames are interpolated linearly; longer gaps stay NaN (on BAR-ATH only 51 frames are filled). The script also computes:
  - velocity by central difference over ±2 frames, masked at block edges;
  - acceleration and turn angle;
  - stillness: the share of frames within ±12 whose speed is below 0.5 m/s;
  - signed distance outside the pitch (`ball_out_m`);
  - gap bookkeeping.
- **Players.** Velocity comes from positions, not the raw speed field (which is missing on about 1.63 tracked slots per frame). Jumps faster than 12.5 m/s are treated as identity switches and set to NaN. Per team the script computes centroid, spread, mean speed, walking share (speed < 2.0 m/s), attack-normalised position, nearest and second-nearest distance to the ball, and the number of players within 3 m and 10 m of the ball.
- **Cross-team.** `nearest_team` (0 = H, 1 = A, −1 = unknown) and `nearest_margin_m` = d_min(A) − d_min(H), which is positive when home is closer. Also free space around the nearest player and all-player speed and walking share.
- **Smart Tagging.** One boolean per (code, team) per frame.
- The IDSSE adapter (`features/external/idsse_features.py`) runs the same `features_from_arrays` code on the IDSSE npz files and adds `label` from DFL BallStatus/BallPossession: `D` where BallStatus is dead, else `H`/`A` from the owner.

### 3.2 Ball-loss simulation on IDSSE (`features/external/degrade_ball.py`)
IDSSE ball tracking is clean, but BAR-ATH loses the ball in long blocks. On BAR-ATH, 132 of 150 missing runs are longer than 5 s, and they hold 99.7% of the missing frames. They behave like stoppages: the players walk, and the ball comes back moving at the restart (findings, 2026-11-09). To tune on data that looks like this, each IDSSE match is degraded, per period, with its own RNG `default_rng([seed, match_index])`:
1. Each DFL dead span [s, e) maps to a ball gap [s + L_on, e + L_re), clipped to the period. Both lags are drawn per span: L_on ~ U{0..100} frames (0–4 s) and L_re ~ U{0..25} frames (0–1 s). A gap of at least 126 frames is dropped (ball made missing) with probability 1.0; a shorter one with probability 0.5.
2. Live-play gaps: a Poisson process at 30 per hour, with lengths log-uniform on [126, 500] frames.
3. Flickers: 10 per hour, with lengths uniform on [1, 12] frames.

All eight values are a priori (decided 2026-11-15) and are not fitted to anything. The live gaps are nonzero on purpose, so that "ball missing" is not a proxy for "dead" and a detector has to use player cues. The lags are random so that the ball's vanish and reappear edges do not sit at a fixed offset from DFL's dead-ball edges. Seeds 0, 1, 2 give 21 degraded tables. The degraded tuning tables show the ball seen 48.1% of the time, with 157.7 runs ≥ 5 s per BAR-ATH-length match (counts scaled to BAR-ATH's 158,857 graded frames), against 59.3% and 132 on BAR-ATH (in-sample descriptive, findings table; nothing was changed in response).

### 3.3 Dead-ball detector v1 (`features/dead_ball.py`)
The design is in `docs/dead-ball-detector-design.md`. The detector is rule-based.

**Player state.** Let W_t be the walking share: the minimum of the two teams' shares, or `all_walking_frac` where a team's share is NaN. Let V_t be the all-player mean speed. Both are smoothed by a NaN-aware centred mean over ±h frames inside the block (h = `dead_smooth_half_window_frames`), giving W̄_t and V̄_t. Then

  stopped_t = [W̄_t ≥ w_min] ∧ [V̄_t ≤ v_max].

A NaN counts as not stopped.

**Candidates.** A frame is D if any of these holds:
- (b) **out of play:** the ball position is known and `ball_out_m` > 0.61 m (ball radius 0.11 m plus 0.5 m tracking error; Law 9);
- (c) **placed ball:** the ball position is known, `ball_still_frac` ≥ 0.5, and stopped_t;
- (a) **long gap:** the ball is unknown for at least `dead_min_gap_frames`, and the gap is not "live". A gap is live when the median of W̄ over the gap is below w_min or the median of V̄ is above v_max. A gap with no finite player data is not called live;
- (d) **known dead:** the published span between the 2nd-half graded start and its kickoff, frames [110808, 113742) on BAR-ATH.

**Gap boundaries.** For a long gap [v, e):
- end = e + `dead_reappear_offset_frames` (= 0): the restart is the first frame the ball is seen again;
- start =
  - v, if the ball was last seen out of play;
  - else onset − `dead_decel_lag_frames`, where onset is the start of the stopped run containing v. If that run reaches back more than `dead_onset_max_lookback_frames` (125), start = v − 125 with no lag;
  - else the first stopped frame inside the gap minus the lag;
  - else v + `dead_vanish_offset_frames` (= 0).

**Clean-up.** Alive runs shorter than `dead_min_alive_frames` between two D candidates in one block are merged into D. D segments shorter than `dead_min_dead_frames` become alive; that value is 0, so this step is off.

**Restart team,** in priority order:
1. a single `SET PIECES` tag team (H/A) overlapping the segment or its restart frame (`N/A` falls through);
2. a `GOALS` tag inside the segment: the conceding team kicks off;
3. the nearest player's team at the restart frame;
4. otherwise "unknown".

The meanings of the Smart Tagging tags are assumptions, not documented facts. The goal-line rule (goal kick vs corner) is deferred to v2, because it needs a last-touch proxy.

The ball vanish and reappear offsets are never tuned. On degraded IDSSE those edges are set by the simulator's lag assumptions, so tuning them would only fit those assumptions.

### 3.4 Possession decoder v1 (`features/possession_decoder.py`)
1. Detector D frames are fixed as `D`; ungraded rows are `X`.
2. Each **alive span** (a run of non-D frames inside one block) is decoded on its own over the states z ∈ {C, H, A}. The per-frame scores are log-likelihood-like.
   - With the ball position known and the margin finite:
     - e_t(H) = clip(m_t / μ, −1, 1), with m_t = `nearest_margin_m` and μ = `decoder_margin_full_m`;
     - e_t(A) = −e_t(H);
     - e_t(C) = c if both teams, or neither, have a player within 3 m of the ball, else 0 (c = `decoder_c_score`).
   - With the ball position or the margin unknown (`ball_unknown`), e_t(C) = 0 and e_t(H) = λ(2p_t − 1), e_t(A) = −e_t(H). Here p_t is the team-shape probability that home has the ball (section 3.5) and λ = `decoder_shape_weight`. At λ = 0 all three scores are 0, which is decoder v1, and the transition costs alone carry the state through the gap.
3. **Transitions.** Costs are π(H↔A) = `decoder_switch_penalty`, π(H/A → C) = `decoder_c_entry_penalty` and π(C → H/A) = `decoder_c_exit_penalty`; staying costs 0. The decoder finds

     ẑ = argmax_z [ b(z_0) + Σ_t e_t(z_t) − Σ_{t≥1} π(z_{t−1}, z_t) ]

   by Viterbi. Exact ties go to the lowest index (C, then H, then A), so flat evidence never invents a team.
4. **Restart bias.** If the span starts at the end of a detector D segment whose restart team is H or A, then b = 0 for that team and b = −`decoder_restart_bias` for the other two states; otherwise b ≡ 0 (Laws 8, 13–17: the restart team plays the ball first). A span at the start of a block gets no bias.
5. **C resolution.** A C run takes the label of the preceding H/A run in the same span, or the following one if there is no preceding run. A loose ball therefore moves a change of possession to where the other team gains control, but never creates a spell of its own. If a span is C throughout, it takes the majority `nearest_team` call over its ball-known frames. If there are none, or they tie, it takes the last H/A label before the span in the block, else the next one after it, and failing that D.

These are the possession rules in football terms:
- **Loose balls and challenges** belong to the team that had the ball until the other team's control outweighs the cost of switching.
- **Restarts** start with the restart team.
- **Ball-unknown live frames** keep the current possession unless the team shape gives enough evidence to switch.

### 3.5 Team-shape model (`features/team_shape.py`, `features/external/fit_team_shape.py`)
This model estimates p(home has the ball) from player shape only, with the ball hidden. It gives the decoder evidence on live frames where the ball position is unknown.
- **Inputs.** For each of eight feature-table columns f, the difference d_f = h_f − a_f:
  - attack-direction normalised per team: `adv_x_m`, `adv_vx`, `own_half_frac`;
  - direction-free: `std_x_m`, `std_y_m`, `spread_m`, `mean_speed`, `walking_frac`.

  No ball-derived column is used, and `test_team_shape.py` asserts this.
- **Model.** An antisymmetric logistic regression without intercept:

    p = σ( Σ_f β_f · d_f / s_f ),

  where s_f is the RMS of d_f over the training rows (scaling only, no centring). Swapping the teams flips every d_f and gives exactly 1 − p, so no home bias can be learned. A NaN feature contributes 0.
- **Fit.** Newton (IRLS) on the penalised log loss −Σ[y log p + (1 − y) log(1 − p)] + (L2/2)·|β|². L2 = 1.0 is fixed a priori: a unit Gaussian prior on each scaled coefficient, which barely shrinks at about 3·10⁵ frames. There is no search and nothing is selected.
- **Training rows.** The alive frames (DFL label H or A) of the clean tables of the 5 tuning matches, with y = 1 for H; 386,329 rows in all. The eight h_/a_ column pairs are asserted to be bit-identical in all 15 degraded tables, so the degradation touches only the ball.
- **Out of fold.** The file also stores 5 leave-one-match-out (LOMO) fold models, each fitted on 4 matches. Decoder tuning run 3 uses them, so no tuning match is scored with a model that saw it. The pipeline itself uses the final model fitted on all 5.
- **Coefficients** (final model, scaled units): adv_x −1.95, adv_vx +0.71, own_half_frac −1.29, std_x +0.23, std_y +1.51, spread +0.34, mean_speed −0.61, walking_frac −0.03. adv_x and own_half_frac are strongly collinear (both say where the play is), so their opposite signs cannot be read separately. Read together, the team in possession is wider (std_y), the play moves toward its attacking goal (adv_vx), and its players run slower on average than the defenders.
- Model, fold models, counts, metrics and data sha256 are in `results/tuning/team_shape/team_shape_model.json`.

`features/make_submission.py` loads the config through `harness.frozen_config()`, decodes the BAR-ATH table, writes `frame,label` and checks the file with the organisers' validator.

## 4. Parameters

All 44 pipeline keys in `features/config.json` are listed below (keys starting with `_` excluded). The config also holds 7 `broadcast_align_*` keys (a priori, diagnostic only, not read by the decoder or the submission; see `docs/broadcast-alignment-design.md`). The full source and reason for each is in `_provenance`. For tuned keys, the tuning data are the IDSSE tuning matches J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ: degraded tables (a priori ball loss), seeds 0–2, 15 runs. The target is the DFL proxy labels via `pm.evaluate`, and the objective is mean spell_f1_mean.

### 4.1 Feature extraction

| key | value | basis | reason (short) | tune log |
|---|---|---|---|---|
| `fps` | `25` | published | Frame rate of video and ATD (games.py, DATA.md). | — |
| `kickoff_video_frames` | `{"1": 11185, "2": 113742}` | published | Kickoff frames (DATA.md, FAQ.md); 2nd half graded from 110808. | — |
| `kickoff_window_frames` | `25` | law_of_game | 1 s window at kickoff to read attack direction (Law 8). | — |
| `kickoff_min_centroid_separation_m` | `5.0` | law_of_game | Loose assertion that the team centroids sit in their own halves at kickoff. | — |
| `kickoff_window_side` | `{"BAR-ATH": "before", "IDSSE": "after"}` | published | Data format: BAR-ATH tracking runs before kickoff; IDSSE starts at it. | — |
| `ball_max_interp_gap_frames` | `12` | a_priori | Interpolate only gaps short enough (0.48 s) to assume no touch. | — |
| `velocity_half_window_frames` | `2` | a_priori | ±2-frame central difference: noise vs time resolution. | — |
| `turn_min_speed_m_s` | `1.0` | physics | Below 1 m/s the ball's direction is lost in jitter. | — |
| `ball_still_speed_m_s` | `0.5` | law_of_game | Restarts are from a stationary ball (Laws 8, 13, 14, 16, 17), with room for jitter. | — |
| `ball_still_half_window_frames` | `12` | a_priori | Stillness over ±0.48 s, so one slow frame in play is not a placed ball. | — |
| `player_max_plausible_speed_m_s` | `12.5` | physics | Above the human sprint record means an identity switch. | — |
| `player_walking_speed_m_s` | `2.0` | published | 7.2 km/h walking/jogging zone boundary (Bradley et al. 2009). | — |
| `near_ball_radii_m` | `[3.0, 10.0]` | law_of_game | 3 m playing distance (design choice); 10 m just above the 9.15 m restart distance. | — |
| `speed_unit_check_moving_m_s` | `2.0` | a_priori | Report-only unit check of the raw speed field. | — |
| `speed_unit_check_tolerance` | `0.1` | a_priori | Report-only unit check tolerance. | — |
| `smart_tagging_team_map` | `{"Barcelona": "H", "Atletic Club": "A", "N/A": "NA"}` | published | Tag team names (DATA.md); N/A kept separate. | — |

### 4.2 Ball-loss simulation (IDSSE only; never applied to BAR-ATH)

| key | value | basis | reason (short) | tune log |
|---|---|---|---|---|
| `ball_loss_dead_drop_prob` | `1.0` | a_priori | User decision: the tracker loses the ball whenever it is dead for ≥ 5 s. | — |
| `ball_loss_short_gap_drop_prob` | `0.5` | a_priori | No prior knowledge for stoppages under 5 s: maximum ignorance. | — |
| `ball_loss_onset_lag_frames` | `[0, 100]` | a_priori | Per-span uniform 0–4 s; the tracker follows the ball briefly after it goes dead. | — |
| `ball_loss_reacquisition_lag_frames` | `[0, 25]` | a_priori | Per-span uniform 0–1 s; not identifiable from allowed data. | — |
| `ball_loss_live_gaps_per_hour` | `30` | a_priori | Live-play occlusions, so "missing" ≠ "dead"; a round guess. | — |
| `ball_loss_live_gap_frames_range` | `[126, 500]` | a_priori | Live gap length 5–20 s, log-uniform; a round guess. | — |
| `ball_loss_flickers_per_hour` | `10` | a_priori | Brief detector dropouts; a round guess. | — |
| `ball_loss_flicker_frames_range` | `[1, 12]` | a_priori | Up to the interpolation limit; a round guess. | — |

### 4.3 Dead-ball detector

| key | value | basis | reason (short) | a priori → tuned | tune log |
|---|---|---|---|---|---|
| `dead_out_margin_m` | `0.61` | physics | Law 9: ball radius 0.11 m plus 0.5 m assumed tracking error. | — | — |
| `dead_placed_still_frac_min` | `0.5` | a_priori | Majority of a symmetric window, so placed-ball edges sit where the ball starts or stops. | — | — |
| `dead_vanish_offset_frames` | `0` | a_priori | Tracker loses the ball when play stops; never tuned (simulator-set edge). | — | — |
| `dead_reappear_offset_frames` | `0` | a_priori | Reappearance = restart; never tuned (simulator-set edge). | — | — |
| `dead_decel_lag_frames` | `0` | tuned_external | Whistle → deceleration lag. Grid edge, at the physical minimum. | 25 → 0 | `results/tuning/dead_ball_tune/tune_log.json` (8df0242a…) |
| `dead_onset_max_lookback_frames` | `125` | a_priori | A whistle comes within seconds; a longer slow stretch is live play. | — | — |
| `dead_smooth_half_window_frames` | `0` | tuned_external | Smoothing window against single-frame noise. Grid edge, at the physical minimum. | 12 → 0 | `results/tuning/dead_ball_tune/tune_log.json` (8df0242a…) |
| `dead_walking_frac_min` | `0.7` | tuned_external | Players stopped: walking share threshold. | 0.8 → 0.7 | `results/tuning/dead_ball_tune/tune_log.json` (8df0242a…) |
| `dead_mean_speed_max_m_s` | `2.0` | tuned_external | Players stopped: mean speed threshold. | 1.5 → 2.0 | `results/tuning/dead_ball_tune/tune_log.json` (8df0242a…) |
| `dead_min_gap_frames` | `50` | tuned_external | Minimum ball gap for a stoppage candidate (2 s). | 50 → 50 | `results/tuning/dead_ball_tune/tune_log.json` (8df0242a…) |
| `dead_min_alive_frames` | `25` | tuned_external | Merge alive blips under 1 s between D candidates. | 25 → 25 | `results/tuning/dead_ball_tune/tune_log.json` (8df0242a…) |
| `dead_min_dead_frames` | `0` | tuned_external | Minimum D length; 0 = off, chosen by the tie rule (all values tied). | (proposed 25) → 0 | `results/tuning/dead_ball_tune/tune_log_min_dead.json` (34880d83…) |

### 4.4 Possession decoder

| key | value | basis | reason (short) | a priori → run 1 → run 2 | tune log |
|---|---|---|---|---|---|
| `decoder_margin_full_m` | `0.25` | tuned_external | Margin at which the team score saturates; a priori one stride. | 2.0 → 1.0 → 0.25 (run-2 edge) | `results/tuning/decoder_tune/tune_log_run2.json` (7672fb2c…) |
| `decoder_c_score` | `0.5` | tuned_external | Score of the contested state on duel or loose-ball frames. | 1.0 → 0.5 | `results/tuning/decoder_tune/tune_log.json` (cebb88d7…) |
| `decoder_switch_penalty` | `50.0` | tuned_external | Direct H↔A cost, about 1 s of clear contrary evidence. Not identified at the chosen entry/exit. | 50 → 50 | `results/tuning/decoder_tune/tune_log.json` (cebb88d7…) |
| `decoder_c_entry_penalty` | `3.0` | tuned_external | H/A → C cost. | 25 → 12 → 3 (run-2 edge) | `results/tuning/decoder_tune/tune_log_run2.json` (7672fb2c…) |
| `decoder_c_exit_penalty` | `25.0` | tuned_external | C → H/A cost. | 25 → 12 → 25 (run-2 edge) | `results/tuning/decoder_tune/tune_log_run2.json` (7672fb2c…) |
| `decoder_restart_bias` | `50.0` | tuned_external | Initial penalty against the non-restart states (Laws 8, 13–17). Not identified at the chosen entry/exit. | 50 → 50 | `results/tuning/decoder_tune/tune_log.json` (cebb88d7…) |
| `decoder_shape_weight` | `2.0` | tuned_external | Weight of the team-shape score on ball-unknown live frames, in units of a clear ball frame. Grid top edge, not extended. | 0.5 → (run 3) 2.0 | `results/tuning/decoder_tune/tune_log_run3.json` (9ee02baf…) |
| `decoder_shape_model` | `"results/tuning/team_shape/team_shape_model.json"` | tuned_external | Team-shape model file: antisymmetric logistic regression, L2 1.0 a priori, fitted by log loss on clean IDSSE tuning tables (alive frames); no search. | fitted, not grid-tuned | `results/tuning/team_shape/team_shape_model.json` (44d8f500…) |

Full sha256 of the tracked logs: `dead_ball_tune/tune_log.json` 8df0242aa07fa1f6acc4904ce26637e552d2abe88a94b003c7bd2dec74fd5e86; `dead_ball_tune/tune_log_min_dead.json` 34880d83e7486b0d4c9889c2fd0bd109bd3ebe46d1ebdfbb7e84cfc8010481f9; `decoder_tune/tune_log.json` cebb88d7214a8033607cf831963549d4e105cda75435c8dbe685193010deb4d5; `decoder_tune/tune_log_run2.json` 7672fb2c559b726595683a52fe220596e40ae6bd46be929e2faad05175ecc6fe; `decoder_tune/tune_log_run3.json` 9ee02baf47b443dabbd5655ff3e954fc8a0d1d7277218e01d186ad9b8ba846c5; `team_shape/team_shape_model.json` 44d8f5008c3f8959fab06e8b390b1f3005f11307733dc07e91330ce4608eaa7b. Also tracked, for the audit trail: `dead_ball_tune/tune_log_run1.json` (52e7be36…) and `dead_ball_tune/crosscheck.json` (56dcd73e…). Both hashes were computed with `shasum -a 256` on 2026-11-16.

The team-shape fit's L2 penalty (1.0, `fit_team_shape.L2_PENALTY`) is a constant in the fitting script, not a config key. Its a priori reason is recorded in the `decoder_shape_model` provenance and in the model file.

## 5. Tuning and evaluation protocol

### 5.1 Harness (`features/external/harness.py`)
- **Scoring.** Scoring is the organisers' `evaluation/possession_metrics.py` (`pm.evaluate`), with nothing re-implemented. The ground truth is the DFL proxy label. The two IDSSE periods sit in adjacent rows, so 26 `X` rows (one more than the widest transition tolerance, 25) are inserted at the seam in both truth and prediction. This keeps every spell and transition match within one half.
- **Leak guard.** Predictors get the table with `label` and `ball_dropped` removed, and the detector raises an error if it sees either.
- **Runs.** Degraded tables give one run per (match, seed), so 15 runs on the tuning split. Clean tables give one run per match. Only `ball_loss_variant = a_priori` tables load.
- **Noise.** We use the organisers' spellF1 noise band of about 0.04 (`SPELL_NOISE_BAND`, EVALUATION.md) to judge differences.

### 5.2 Declared objective and tie rules
Each tune log records its grid, a priori point, fixed keys, seeds, tuning ids, objective and tie rule before the first score. The objective is always to maximise mean spell_f1_mean over the 15 degraded tuning runs. Tie rules:
- **Dead-ball joint grid:** among combos within 0.002 of the best, take the highest mean alive_dead_f1; then the smallest grid-index distance to the a priori point; then grid order.
- **`dead_min_dead_frames`:** if the top values are within 0.001 of each other, take the smallest value.
- **Decoder:** among combos within 0.002 of the best, take the smallest grid-index distance to the a priori point; then grid order.

### 5.3 Runs
| run | log | keys | combos | chosen |
|---|---|---|---|---|
| dead-ball run 1 (2026-11-15T23:58) | `tune_log_run1.json` | six player keys | 10,000 | lag 12, smoothing 6 (edge), walking 0.7, speed 2.0, min_gap 50, min_alive 0 (edge) |
| dead-ball run 2, edge extension (2026-11-16T01:29) | `tune_log.json` | same; smoothing grid extended once to {0, 3} | 15,000 | lag 0 (edge), smoothing 0 (edge), walking 0.7, speed 2.0, min_gap 50, min_alive 25 |
| `dead_min_dead_frames` (2026-11-16T10:21) | `tune_log_min_dead.json` | one key over {0, 12, 25, 38, 50} | 5 | 0 |
| decoder run 1 (2026-11-16T10:33) | `decoder_tune/tune_log.json` | six decoder keys, 3 values each | 729 | margin 1.0, c_score 0.5, switch 50, entry 12, exit 12, restart_bias 50 |
| decoder run 2, edge extension (2026-11-16T23:03) | `decoder_tune/tune_log_run2.json` | margin {0.25, 0.5, 1.0, 2.0}, entry and exit {3, 6, 12, 25} | 64 | margin 0.25, entry 3, exit 25 |
| team-shape fit (2026-11-16T23:22) | `team_shape/team_shape_model.json` | logistic coefficients, by log loss; no grid | — | final model + 5 LOMO fold models |
| decoder run 3, team-shape weight (2026-11-16T23:23) | `decoder_tune/tune_log_run3.json` | `decoder_shape_weight` over {0, 0.125, 0.25, 0.5, 1.0, 2.0}, all other keys at run-2 values | 6 | 2.0 (top edge) |

**Run 3.** The a priori value 0.5 and the grid (`GRID_RUN3`) were declared in `tune_decoder.py` before the team-shape fit and before the run. Run 3 uses the same tables, seeds, objective and tie rule as runs 1–2. It records the run-2 log as `prior_log`, and the model file's path and sha256 as `shape_model`. Each tuning match gets p_home from its LOMO fold model (asserted: the fold must not have seen that match). The weight 2.0 is on the top edge of the grid. It was not extended, because findings records that it will not be extended without a user decision.

**Grid-edge rule: one extension only.** When a chosen value sat on a grid edge, the grid was extended once and the result was reported, not extended again.
- **Dead-ball run 2.** The only extension was the smoothing grid. min_alive could not be extended, because 0 is its hard lower bound. Lag 0 and smoothing 0 are both edges and physical minimums.
- **Decoder run 2.** All three chosen values are edges of the run-2 grid. 0.25 m and 3 are near the physical floor of 0. Exit 25 is not a physical bound, and exit 50 was never tried together with entry 3 or 6. Run 2 recorded the run-1 log as `prior_log`. The run-1 point reproduced exactly in run 2 (0.38136).
- **Why run 2 is kept over run 1.** It rests on the declared procedure, not on evidence of a better optimum: the gain is +0.009, inside any reasonable noise band.

### 5.4 Enforcement
- `harness.frozen_config()` is the only way the tuned pipeline loads its config (harness predictors, `make_submission.py`). It runs `check_provenance`, then checks the following against the logs:
  - every tuned key names a tune log that `frozen_config` actually checks;
  - the config equals the chosen rows of both dead-ball logs;
  - the decoder keys not re-tuned in run 2 equal the run-1 log;
  - the run-2 keys equal the run-2 log;
  - the run-2 log's `prior_log` sha256 equals the run-1 log;
  - `decoder_shape_weight` equals the run-3 log, and the run-3 log's `prior_log` is the run-2 log;
  - `decoder_shape_model` is the tracked model file, and its sha256 equals the one run 3 was tuned with.
- `tune_dead_ball.py --check --log <log>` prints the detector check on its own. There is no separate `--check` for the decoder: the decoder logs are checked only through `harness.frozen_config()`, which runs on every `make_submission.py` call and every harness `decoder_v1` run.
- The tune logs are tracked in `results/tuning/`. Each `tune_log` provenance entry carries a path and sha256 that `check_provenance` verifies on every run. The logs were moved there from `external_data/` unchanged (same sha256). Their internal `previous_run` / `prior_log` / `tune_log` fields still name the old `external_data/` paths.
- The logs also record the sha256 of the script, harness, detector and decoder, and run 3 also records the team-shape module. Checked against the current files on 2026-11-16:
  - `dead_ball.py` matches every log.
  - `harness.py`, `tune_decoder.py` and `team_shape.py` match the run-3 log and the model file. `possession_decoder.py` does not: its docstring was corrected after run 3 (text only; the C-score wording for ball-known, margin-NaN frames).
  - The hashes logged for `harness.py` and `tune_decoder.py` in the dead-ball logs and decoder runs 1–2 no longer match, and `tune_dead_ball.py` matches none of its logs. Later edits explain this: the `frozen_config` checks for runs 2 and 3, docstring log paths updated when the logs moved, the `--run3` option, and p_home passed to the decoder.
  - The search code of runs 1–2 is unchanged (findings). At weight 0 the run-3 code reproduces run 2's 0.39054 exactly.

## 6. Results

All IDSSE numbers below are mean (min–max) over runs, scored by `pm.evaluate` against DFL proxy labels. Each table says whether it is in-sample or held-out.

### 6.1 Tuning split (in-sample), detector stage

| predictor | tables | runs | spell_f1_mean | spell_f1@0.5 | macro_f1 | alive_dead_f1 | team_acc_gt_alive | n_pred_spells | fragmentation |
|---|---|---|---|---|---|---|---|---|---|
| baseline_nearest | degraded | 15 | 0.106 (0.08–0.13) | 0.092 (0.07–0.11) | 0.720 (0.69–0.75) | 0.861 (0.81–0.90) | 0.617 (0.55–0.67) | 1765 (1529–1994) | 3.45 (2.78–4.25) |
| dead_ball_v1 untuned | degraded | 15 | 0.120 (0.09–0.14) | 0.106 (0.08–0.13) | 0.756 (0.73–0.80) | 0.917 (0.90–0.95) | 0.730 (0.70–0.75) | 1704 (1471–1953) | 3.26 (2.60–4.11) |
| dead_ball_v1 tuned | degraded | 15 | 0.121 (0.09–0.14) | 0.107 (0.08–0.13) | 0.787 (0.77–0.81) | 0.949 (0.93–0.97) | 0.722 (0.69–0.75) | 1715 (1486–1974) | 3.31 (2.65–4.20) |
| baseline_nearest | clean | 5 | 0.100 (0.08–0.12) | 0.086 (0.07–0.10) | 0.359 (0.33–0.38) | 0.704 (0.65–0.74) | 0.764 (0.75–0.78) | 2092 (1981–2240) | 3.77 (3.32–4.43) |
| dead_ball_v1 untuned | clean | 5 | 0.115 (0.09–0.14) | 0.101 (0.08–0.12) | 0.776 (0.76–0.81) | 0.933 (0.92–0.96) | 0.757 (0.74–0.77) | 2124 (2014–2271) | 4.09 (3.55–4.84) |
| dead_ball_v1 tuned | clean | 5 | 0.117 (0.09–0.14) | 0.103 (0.08–0.13) | 0.804 (0.78–0.83) | 0.961 (0.95–0.98) | 0.754 (0.74–0.77) | 2108 (1988–2264) | 4.06 (3.52–4.85) |

The tuning matches have 454 (401–511) true spells per match. In the detector-stage tables, "nearest team" fills the alive frames: the nearest player's team, carried forward over unknown-ball frames.
- The detector lifts alive/dead to alive_dead_f1 0.95 (in-sample, degraded). On clean tables the ball is never missing, yet alive_dead_f1 is 0.96, so the player cues do the work there.
- spellF1 barely moves. The predictions are 3–4 times more fragmented than the truth, because the nearest-player identity flips during passes and duels. That is what the decoder is for.
- **Flat surface.** In dead-ball run 2 (in-sample), mean spell_f1 spans only 0.111–0.122 across all 15,000 combos, and 1,768 combos are within the 0.002 tie band. The choice was in practice made by alive_dead_f1. Chosen combo: spell_f1 0.1212, alive_dead_f1 0.9494. Only the walking and speed thresholds move alive_dead_f1 much.
- **Clean cross-check** (in-sample, `crosscheck.json`). Three keys are flagged: smoothing, min_alive and walking. Only walking matters in practice: clean alive_dead_f1 is 0.961 at 0.7, against 0.973 at 0.6, which is the grid edge. The chosen values were not changed.

### 6.2 Tuning split (in-sample), decoder stage (degraded, 15 runs)

| config | spell_f1_mean | macro_f1 | team acc (GT alive) | pred spells / match | true spells / match |
|---|---|---|---|---|---|
| dead_ball_v1 + nearest team | 0.121 | 0.787 | 0.722 | 1715 | 454 |
| decoder, a priori values | 0.277 | — | — | 181 | 454 |
| decoder, run 1 | 0.381 | 0.827 | 0.781 | 289 | 454 |
| decoder, run 2 | 0.391 | 0.830 | 0.785 | 275 | 454 |
| decoder + team shape, run 3 (current) | 0.418 | 0.842 | 0.804 | 300 | 454 |

Best in-sample spell_f1 per value of each key, other keys free:
- **Run 1.**
  - margin 1.0 / 2.0 / 4.0: .381 / .365 / .349;
  - c_score 0 / 0.5 / 1.0: .372 / .381 / .345;
  - switch: .381 for every value;
  - entry, and likewise exit, 12 / 25 / 50: .381 / .372 / .363;
  - restart_bias 0: .375, 50 and 100: .381.
- **Run 2.**
  - margin 0.25 / 0.5 / 1.0 / 2.0: .391 / .388 / .382 / .372;
  - entry 3 / 6 / 12 / 25: .391 / .385 / .388 / .386;
  - exit 3 / 6 / 12 / 25: .386 / .382 / .388 / .391.

The good combos lie on a ridge where entry + exit ≈ 24–28. The surface pins the total cost of going through C, not how it is split.

The decoder cuts fragmentation from 1715 to 275 predicted spells per match (run 2), against 454 true (in-sample). It now under-segments. The team-shape score in run 3 raises this to 300.

### 6.3 Team-shape model and decoder run 3 (in-sample on the tuning matches; LOMO is out-of-match)

**Leave-one-match-out accuracy and log loss of the team-shape model.** Each model is fitted on 4 tuning matches and predicts the 5th. Accuracy uses p > 0.5 → H; the second number is mean log loss.

| match | alive frames, clean | ball-unknown alive frames, degraded (3 seeds) |
|---|---|---|
| J03WMX | 0.841 / 0.377 | 0.856 / 0.350 |
| J03WN1 | 0.758 / 0.511 | 0.756 / 0.518 |
| J03WOH | 0.838 / 0.372 | 0.854 / 0.341 |
| J03WOY | 0.826 / 0.393 | 0.863 / 0.335 |
| J03WQQ | 0.841 / 0.350 | 0.852 / 0.347 |
| pooled | 0.823 / 0.397 (n 386,329) | 0.837 / 0.377 (n 217,375) |

These are out-of-match within the tuning split: the held-out matches were not used.

**Run-3 profile** (degraded, 15 runs, in-sample; p_home from the LOMO fold models):

| decoder_shape_weight | 0 | 0.125 | 0.25 | 0.5 (a priori) | 1.0 | 2.0 (chosen) |
|---|---|---|---|---|---|---|
| spell_f1_mean | 0.3905 | 0.3990 | 0.4042 | 0.4113 | 0.4151 | **0.4178** |
| team acc (GT alive) | 0.785 | 0.796 | 0.800 | 0.802 | 0.803 | 0.804 |
| pred spells / match | 275 | 278 | 283 | 289 | 294 | 300 |

- Weight 0 reproduces run 2 exactly (0.3905).
- **Edge.** The chosen value, 2.0, is on the top edge of the grid. 1.0 is 0.0026 below, just outside the 0.002 tie band. The curve is flattening: +0.004, then +0.003 per doubling.
- **Size of the gain.** The gain from the a priori 0.5 to 2.0 is 0.006, small next to the per-run spread (0.39–0.47). The gain from run 2 is +0.027 in-sample.
- **Fold models vs final model.** An ad-hoc harness `decoder_v1` run on the tuning tables, using the final all-5 model in place of the fold models, also gave 0.418 (not logged, not used for any choice). Out-of-fold and in-sample shape probabilities score about the same here.

### 6.4 Held-out (J03WPY, J03WR9), evaluation only
No integrated-pipeline held-out number exists yet (TODO, section 8). The looks already logged are below. Nothing was tuned or selected on them.

| predictor | tables | spell_f1_mean | macro_f1 | alive_dead_f1 | team_acc_gt_alive | n_pred_spells |
|---|---|---|---|---|---|---|
| baseline_nearest (fixed-lag degraded tables, not re-run) | degraded (6 runs) | 0.103 | 0.757 | 0.896 | 0.681 | 1792 |
| baseline_nearest | clean (2 runs) | 0.095 | 0.376 | 0.725 | 0.797 | 1991 |
| dead_ball_v1 tuned (look 1) | degraded (6 runs) | 0.115 | 0.797 | 0.950 | 0.746 | 1681.5 |
| dead_ball_v1 tuned (look 1) | clean (2 runs) | 0.113 | 0.820 | 0.965 | 0.784 | 2012 |

The held-out pair has 396 (370–421) true spells per match. The 2012 for dead_ball_v1 clean is from `holdout_log.json`; findings lists it as n/a.
- **Held-out alive/dead.** alive_dead_f1 on held-out degraded tables is 0.950, against 0.949 in-sample: the alive/dead split transfers across matches. Held-out spell_f1 (0.115) lies inside the flat in-sample band, so it says nothing about overfitting.
- **Look 2, D-segment counts** (`dead_segment_counts.py`). Held out: 126.5 predicted vs 94.5 true D segments per match on degraded tables, and 147.5 vs 94.5 on clean ones. Predicted segments shorter than 1 s are 16.8% (degraded) and 18.8% (clean). Predicted D share is 42.6% vs 43.1% true on degraded tables. The in-sample tuning matches show the same pattern: 119.5 and 143.6 vs 104.8.

### 6.5 BAR-ATH, label-free plausibility (no ground truth, no video)
- **Dead-ball detector** (`features/dead_ball_report.py`, tuned detector):
  - predicted D share 43.1% of graded frames, or 42.0% excluding the published 2nd-half pre-kickoff span. That is inside the range of predicted D shares on the IDSSE tuning matches (42.4% clean to 44.5% degraded, split means). With the held-out split means included, the range is 40.6–44.5%.
  - 145 D segments (65 in the 1st half, 80 in the 2nd), 144 excluding the pre-kickoff span. That is more than the IDSSE true mean of 104.8 per match (range 99–113), as the detector also over-counts on IDSSE.
  - 13 segments (9.0%) are shorter than 1 s.
  - Segment length quartiles are 2.7 / 9.8 / 21.6 s; the longest is 221.2 s.
  - Restart team: A 76, H 65, unknown 4.
  - 78 of 145 segments overlap a set-piece tag (an upper bound).
  - 16.8% of D frames have the ball seen; 88.1% of ball-missing graded frames are called D.
- **Submission label shares** (`make_submission.py`, validator OK, label-free). The first row is given on the same base as the others (graded frames, 158,857). It was converted from findings' all-frame shares (199,319 frames), so its values carry about ±0.1 pp rounding.

| config | H | A | D | X (ungraded frames) | sha256 |
|---|---|---|---|---|---|
| decoder run 1 (history) | 33.2% | 23.7% | 43.0% | 40,462 | not recorded |
| decoder run 2 (history) | 51,439 (32.4%) | 38,984 (24.5%) | 68,434 (43.1%) | 40,462 | d47ab2db… |
| **decoder run 3, team shape (current)** | **53,421 (33.6%)** | **37,002 (23.3%)** | **68,434 (43.1%)** | 40,462 | **17511abf…** |

- **Run 1.** findings gives all-frame shares H 26.5%, A 18.9%, D 34.3%, X 20.3%. The validator counted 426 possessions with a mean of 8.5 s; the a priori decoder gave 224 of 16.1 s.
- **Run 2.** A fresh clone running `sh reproduce.sh` reproduced its file byte for byte (findings, clean-clone test at commit 80a2541).
- **Run 3.** D is unchanged from run 2, because the team-shape model only re-scores alive frames. On net 1,982 graded frames moved from A to H. The label-free D-segment report is unchanged (145 segments).

## 7. Limitations and risks

- **Transfer from clean IDSSE player tracking.** The degraded tables degrade only the ball; IDSSE player positions stay clean. Several tuned values sit exactly where clean and noisy tracking would differ most:
  - dead-ball smoothing 0 and lag 0;
  - walking 0.7;
  - decoder margin 0.25 m, which turns the team score into close to the sign of a noisy margin. The config itself assumes about 0.5 m ball error.

  Treat these values as weakly supported on BAR-ATH.
- **Parameters at grid edges.**
  - Detector: lag and smoothing are at the edge, both at a physical minimum.
  - Decoder: all three run-2 keys are at the edge. c_exit 25 is not a physical bound, and the optimum may lie beyond it.
  - Team-shape weight: 2.0 is at the top edge.

  For the detector and decoder keys the single-extension rule stopped further search; the weight was not extended without a decision.
- **Unidentified parameters.** At the chosen entry/exit costs, a change of team routed through C costs at most about 28 (entry + exit), less than the switch penalty of 50. So switch_penalty and restart_bias have no effect at the chosen values. Their values rest on their a priori reasons only.
- **Flat, unstable dead-ball surface.** spell_f1 does not identify the detector keys. Between run 1 and run 2, lag went 12 → 0 and min_alive 0 → 25, although only the smoothing grid was extended.
- **The ball-loss model is an assumption.** The drop probabilities, the lag spreads (0–4 s onset, 0–1 s reacquisition) and the live-gap and flicker rates are a priori. Degraded alive/dead scores are only as realistic as those spreads. The reacquisition lag cannot be identified from any data we may use. The simulator does not match BAR-ATH exactly: the ball is seen 48.1% of the time vs 59.3%, the long-gap median is 15.1 s vs 11.6 s, and it produces too few very long gaps (findings table). The a priori values were chosen after BAR-ATH's tracking aggregates had been seen, so they are not blind.
- **Proxy labels.** DFL's alive/dead timing and possession hand-over (BallStatus/BallPossession) may follow a different convention from the organisers' ground truth (whistle vs line crossing; touch vs control). All boundary placement learned on IDSSE inherits DFL's convention.
- **Over-segmented D.** The detector predicts more D segments than the truth: 14–37% more on the IDSSE tuning matches, and 144 on BAR-ATH against the IDSSE tuning-match mean of 104.8. A minimum D length did not help (all values tied).
- **Under-segmented possessions.** The decoder predicts 300 spells per match (run 3) against 454 true (in-sample). Short possessions are being merged.
- **Restart team.** The Smart Tagging tag meanings are assumptions. Goal kick vs corner is not decided by geometry yet (v2), and throw-ins fall back to the nearest player.
- **Team-shape transfer, IDSSE to BAR-ATH.** The model learned Bundesliga team shapes from clean player tracking. BAR-ATH has noisier tracking and different teams. A possession style like Barcelona's (long, wide home possessions) may change the map from shape to possession. Antisymmetry removes home bias, but not a team-style bias.
- **Team-shape weight.** 2.0 sits on the top edge of the grid and was not extended. At 2.0, a confident shape frame counts as much as two clear ball frames. The a priori rationale for 0.5 (about 4–5 s of consistent shape evidence to switch inside a ball gap) no longer holds: the needed evidence shrinks roughly in proportion to the weight. That last point is our reading of the score scale, not a logged measurement.
- **Where the team-shape model acts.** It only matters on ball-unknown live frames. On IDSSE these come from the a priori degradation, and their share and placement on BAR-ATH may differ. On BAR-ATH most ball-missing frames are stoppages that the detector calls D (11.9% of ball-missing frames are called alive), so the model acts only on that remainder.
- **Collinear shape features.** adv_x and own_half_frac are strongly collinear, so individual coefficients are not interpretable, though the prediction is unaffected.
- **Held-out pair.** It is not pristine: it has had one baseline look and two detector looks (section 2.4). The one final evaluation will be reported as a further look.

## 8. Pending work

- **TODO: broadcast alignment.** Not started in this draft. Placeholder: what it adds (ball position in tactical-camera gaps), the method, and how it complies with the rules (no checking output against the video).
- **TODO: final held-out evaluation.** One run of the integrated pipeline on J03WPY and J03WR9 (degraded, seeds 0–2, plus clean), with `--holdout` and logged in `holdout_log.json`. Nothing is changed after it. Placeholder for the results table and a short discussion.
- **TODO:** if the config changes again (for example after broadcast alignment), re-run `make_submission.py` and `dead_ball_report.py` and update section 6.5. The run-3 aggregates are recorded there.
- **TODO:** the team-shape model is under review and not yet committed. Update this document if the review changes it.

## 9. How to rerun

### 9.1 BAR-ATH submission
From a fresh clone, with the provided data unpacked into `data/`:

    sh reproduce.sh

It was developed and tested with Python 3.14.0. `reproduce.sh` calls `python3` and does not pin the interpreter version. The script does the following:
1. creates `.venv` and installs the pinned `features/requirements.txt` (numpy 2.5.3, pandas 3.0.5, pyarrow 26.0.0);
2. runs `features/extract_features.py`, which writes `data/derived/features_tactical.parquet`;
3. runs `features/make_submission.py`. That script loads the config through `frozen_config()`, which checks it against the tracked tune logs and their sha256, then decodes and writes `data/derived/submission.csv`. It validates the file with the official validator, and prints `wrote … (199319 rows, valid)` and the H/A/D share of graded frames plus the X count;
4. prints `shasum -a 256 data/derived/submission.csv`. For the current run-3 config, findings records sha256 17511abf…. TODO: give the full hash of the file actually submitted.

It needs no IDSSE data, no tuning and no network access beyond pip. It stops with an error if the config does not match the logs. A label-free sanity summary is available from `python features/extract_features.py --report`.

### 9.2 IDSSE pipeline (external data, optional)
These steps rebuild the tuning inputs in `external_data/` (gitignored). Both requirement files are needed, because the IDSSE steps import the feature extractor:

    pip install -r features/requirements.txt -r features/external/requirements.txt   # + kloppy 3.19.1, lxml
    python features/external/fetch_idsse.py --all                  # -> external_data/idsse/*.npz
    python features/external/idsse_features.py --all               # -> external_data/idsse_features/
    python features/external/degrade_ball.py --all --seeds 0 1 2   # -> external_data/idsse_degraded/

    # baseline, detector and full decoder on the tuning split (prints a table, writes nothing)
    python features/external/harness.py --predictor baseline_nearest --tables degraded --seeds 0 1 2
    python features/external/harness.py --predictor dead_ball_v1 --tables degraded --seeds 0 1 2
    python features/external/harness.py --predictor decoder_v1 --tables degraded --seeds 0 1 2

    # check the config against the tracked dead-ball logs (read-only)
    python features/external/tune_dead_ball.py --check --log results/tuning/dead_ball_tune/tune_log.json
    python features/external/tune_dead_ball.py --check --log results/tuning/dead_ball_tune/tune_log_min_dead.json

The tuning runs can be repeated into a **new** directory, so the tracked logs are never touched. Every tune and fit script opens its output with mode `"x"` and refuses to overwrite an existing file. Here `R=/tmp/rerun` stands for any empty directory:

    R=/tmp/rerun; mkdir -p $R
    # dead-ball run 1, then run 2 = edge extension of the smoothing grid
    python features/external/tune_dead_ball.py --tune --log $R/db_run1.json --workers 7
    python features/external/tune_dead_ball.py --tune --log $R/db_run2.json --workers 7 \
        --extend dead_smooth_half_window_frames=0,3 --previous results/tuning/dead_ball_tune/tune_log_run1.json
    python features/external/tune_dead_ball.py --crosscheck --log results/tuning/dead_ball_tune/tune_log.json \
        --out $R/crosscheck.json --workers 7
    python features/external/tune_dead_ball.py --tune-one dead_min_dead_frames --log $R/db_min_dead.json --workers 5
    # team-shape fit, then decoder runs 1, 2 and 3
    python features/external/fit_team_shape.py --out $R/team_shape_model.json
    python features/external/tune_decoder.py --log $R/dec_run1.json --workers 7
    python features/external/tune_decoder.py --run2 --log $R/dec_run2.json --workers 7
    python features/external/tune_decoder.py --run3 --log $R/dec_run3.json --workers 7

Caveats:
- These commands follow the scripts' argument parsers. They were not re-run for this document.
- `--run2` and `--run3` check the config against the tracked run-1 and run-2 logs (`prior_log`), and the decoder runs read the tracked team-shape model named in the config. So a rerun repeats each search against the tracked inputs, not against the rerun's own outputs.
- The new logs record current script hashes, which differ from the logged ones (section 5.4). Compare the `chosen` and `results` fields, not the file sha256.
- Held-out runs need `--holdout` and are logged. Under the policy in section 2.4, only the final evaluation may use it.

### 9.3 Tests
Synthetic-match tests, run after any change to the extractor (the command from CLAUDE.md, now including `test_team_shape.py`):

    python features/tests/make_synthetic_match.py /tmp/synth && MTAH_DATA_DIR=/tmp/synth python features/tests/test_synthetic.py && python features/tests/test_fetch_checks.py && python features/tests/test_idsse_features.py && python features/tests/test_degrade_ball.py && python features/tests/test_harness.py && python features/tests/test_dead_ball.py && python features/tests/test_tune_dead_ball.py && MTAH_DATA_DIR=/tmp/synth python features/tests/test_possession_decoder.py && python features/tests/test_team_shape.py

## 10. Credits and licences

- **IDSSE data.** Bassek, Rein, Weber et al. "An integrated dataset of spatiotemporal and event data in elite soccer." *Scientific Data* 12, 195 (2025). doi:10.1038/s41597-025-04505-y. Licence CC BY 4.0. Downloaded from huggingface.co/datasets/pysport/idsse-data. We used the 7 matches J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ, J03WPY and J03WR9, converted to compact arrays. The raw XML was not kept.
- **kloppy** (version 3.19.1, BSD-3-Clause licence): IDSSE loading, and the URL pattern and orientation check in `fetch_idsse.py` / `idsse_features.py`.
- **The organisers' evaluation code** (`evaluation/possession_metrics.py`, `validate_submission.py`; MIT licence, copyright Metrica Sports). It is the only scorer used, both on IDSSE (`pm.evaluate`) and to validate the submission.
- **Provided match data** (BAR-ATH tactical ATD and Smart Tagging): used under the competition's terms of use, locally only, and not redistributed.
- Published references for parameters: IFAB Laws of the Game (Laws 2, 8, 9, 13–17); Bradley et al. 2009 (speed zones); Bohannon 1997 (walking speed). The full list is in `features/config.json` `_provenance`.
