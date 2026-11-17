# MTAH 2027: possession per frame on BAR-ATH (technical document)

Version of 2026-11-17. Every number below comes from `docs/findings.md`, `features/config.json`, a tracked tune log in `results/tuning/`, or the held-out log (`external_data/harness/holdout_log.json`, tracked copy `results/holdout/holdout_log.json`). Each number says whether it is in-sample or held-out. No number here was computed on BAR-ATH ground truth or checked against the video. The main body is short; every parameter is in Appendix A, the tuning detail in Appendix B and the rerun commands in Appendix C.

## 1. Summary

**Task.** Label every frame of the BAR-ATH tactical video (199,319 frames, 158,857 graded) as `H` (home in possession), `A` (away in possession), `D` (dead ball) or `X` (ungraded). The ranking metric is spellF1: each possession, a maximal run of `H` or `A`, is matched one-to-one against the truth by temporal IoU, and the F1 is averaged over IoU thresholds 0.3–0.7.

**Approach.** The pipeline uses only the provided tactical tracking (ATD) and Smart Tagging. (1) A feature extractor turns the raw ATD into one row of ball, team and cross-team features per frame. (2) A rule-based dead-ball detector marks the stoppages. On BAR-ATH the tracker loses the ball in long blocks, mostly while it is dead, so a long ball gap together with players walking is the main dead-ball cue; ball-out-of-play and placed-ball cues are added, and Smart Tagging names the team that restarts. (3) A possession decoder runs a three-state Viterbi (home, away, contested/loose) over each alive span, scoring each frame by the nearest-player margin and charging a penalty for each change of state. When the ball position is unknown, a team-shape model scores the frame instead: a logistic regression on home-minus-away player-shape differences, fitted on IDSSE with the ball hidden. Contested frames take the preceding team's label, so a loose ball never splits a spell. Every parameter is published, follows from the Laws of the Game or physics, was chosen a priori, or was tuned on an external dataset (IDSSE, the 7 Bundesliga matches of Bassek et al. 2025), whose tuning tables were first degraded with a simulated BAR-ATH-style ball loss.

**Key numbers. In-sample, on the five IDSSE tuning matches with degraded ball tracking, 15 runs (5 matches × 3 seeds), scored with the organisers' `pm.evaluate` against DFL proxy labels.**

| pipeline | spell_f1_mean (in-sample) |
|---|---|
| nearest-player baseline | 0.106 |
| dead-ball detector + nearest team (tuned) | 0.121 |
| + possession decoder, tuning run 1 | 0.381 |
| + possession decoder, tuning run 2 | 0.391 |
| + team-shape score on ball-unknown frames, tuning run 3 (submitted config) | 0.418 |

**Held-out.** One final, logged evaluation of the integrated pipeline on the held-out IDSSE matches J03WPY and J03WR9 (6 runs, degraded): spell_f1_mean **0.417** (0.399–0.434), against 0.418 in-sample (section 6.4). Nothing was changed after it.

## 2. Data and rules compliance

- **Provided data stays local.** The BAR-ATH tracking, Smart Tagging and video are used only on the local machine; the derived feature table and the submission go to `data/derived/`, which is gitignored and never committed or uploaded. An organisers' clarification of 2026-11-03 forbids uploading the provided data to AI services. AI coding assistants were used for this entry; they never read rows of anything under `data/`. Scripts ran locally and the assistant saw only their aggregate summaries; `.claude/settings.json` blocks direct reads of `data/`.
- **Assets used:** the tactical ATD (player and ball positions) and Smart Tagging (only `SET PIECES` and `GOALS`, only to name the restart team). **Not used in the submission:** the broadcast video and broadcast tracking (section 5), the homography, and the undocumented third raw ball field `ball_raw_f2`.
- **No fitting on BAR-ATH.** Nothing was trained, fitted or tuned on the BAR-ATH ground truth or the video, including the public 5-minute ground-truth window (`docs/findings.md` records no score on it). There are no hand labels on BAR-ATH, and the output was never checked against the video.
- **"Not blind" disclosure.** The BAR-ATH checks we ran are label-free aggregates of tracking and Smart Tagging (D share, D segments, gap statistics). No value was fitted to them, but they had been seen before these choices:
  - **Ball-loss simulator.** An earlier version was fitted to BAR-ATH's label-free tracking aggregates (never to GT). On 2026-11-15 the user decided not to use it until the organisers confirm such fitting is allowed; the fitted values are archived, unused, in `features/external/ball_loss_fitted_bar_ath.json`. The active values are a priori, but chosen by people who had seen those aggregates and the fitted values.
  - **Dead-ball thresholds.** The a priori points of `dead_walking_frac_min` (0.8) and `dead_mean_speed_max_m_s` (1.5 m/s) were set after the BAR-ATH gap medians had been seen (walking share 0.95 and mean speed 1.00 m/s in long gaps, against 0.62 and 1.93 m/s with the ball seen). The tie rule falls back on grid-index distance to the a priori point, so these aggregates could influence the tuned choice indirectly, though only on exact ties in alive_dead_f1; in practice the choice was made by alive_dead_f1 (Appendix B).
  - **`dead_min_dead_frames`** was proposed after the label-free BAR-ATH report (13 of 145 D segments under 1 s) and after the held-out D-segment look. It was tuned on the tuning matches only and came out at 0 (off).
- **Provenance enforced.** Every config key has a `_provenance` entry with `basis` ∈ {`published`, `law_of_game`, `physics`, `a_priori`, `tuned_external`}, plus `source`, `reason` and `tuned_on`. `extract_features.check_provenance()` refuses to run if a key has no entry (or an entry no key); a `tuned_external` key names no dataset or target, names BAR-ATH, or names a held-out IDSSE match as tuning data; or a `tune_log` lies outside `results/tuning/`, is missing, or has a different sha256 from the recorded one.

## 3. Method

The code is in `features/`. Every stage works per `half` block: nothing (smoothing, runs, merging, decoding) crosses a half edge or, on IDSSE, the period seam. All intervals are half-open.

**3.1 Feature extraction** (`features/extract_features.py`). The raw ATD is streamed into arrays [frames × slots × (x, y, speed)] plus ball fields; any unexpected format is an error. Coordinates are scaled by the published pitch (105.3 × 68.0 m); the parse matches DATA.md (ball seen 59.3%, 20.49 tracked players). Each team's attack direction per half is read off the kickoff formation (Law 8). Ball gaps up to 12 frames are interpolated; longer gaps stay NaN. Per frame the extractor computes ball velocity, acceleration, turn angle, stillness, distance outside the pitch and gap bookkeeping; per team centroid, spread, mean speed, walking share, attack-normalised position and distances to the ball (player velocities from positions, jumps above 12.5 m/s treated as identity switches); and the cross-team `nearest_team` and `nearest_margin_m` = d_min(A) − d_min(H). The IDSSE adapter (`features/external/idsse_features.py`) runs the same code on IDSSE and adds the DFL label.

**3.2 Ball-loss simulation on IDSSE** (`features/external/degrade_ball.py`). IDSSE ball tracking is clean; BAR-ATH loses the ball in long blocks (132 of 150 missing runs are over 5 s and hold 99.7% of missing frames; they behave like stoppages). Each IDSSE match is degraded per period: each DFL dead span becomes a ball gap with random onset (0–4 s) and reacquisition (0–1 s) lags, dropped with probability 1.0 if ≥ 5 s, else 0.5; plus live-play gaps (30 per hour, 5–20 s) so that "missing" is not a proxy for "dead", and short flickers. All eight values are a priori (2026-11-15). Seeds 0–2 give 21 degraded tables. The degraded tuning tables show the ball seen 48.1% of the time, against 59.3% on BAR-ATH (descriptive; nothing changed in response).

**3.3 Dead-ball detector v1** (`features/dead_ball.py`, design in `docs/dead-ball-detector-design.md`). Players are "stopped" when the (smoothed) walking share is at least w_min and the mean speed at most v_max. A frame is D if (b) the ball is known and more than 0.61 m out of play (Law 9); (c) the ball is known, still, and players are stopped (placed ball); (a) the ball is unknown for at least `dead_min_gap_frames` and the gap is not live by its player medians; or (d) it lies in the published span between the 2nd-half graded start and kickoff. A gap's D start is moved back to the onset of the stopped run (bounded to 5 s); its end is the frame the ball reappears. Short alive blips between D candidates are merged. The restart team comes from a `SET PIECES` tag, else a `GOALS` tag (the conceding team kicks off), else the nearest player at restart. Tag meanings are assumptions; the ball vanish/reappear offsets are never tuned, because on degraded IDSSE those edges are set by the simulator.

**3.4 Possession decoder v1** (`features/possession_decoder.py`). Detector D frames are fixed; each alive span is decoded on its own over states {C, H, A}. With the ball and margin known, e(H) = clip(m/μ, −1, 1), e(A) = −e(H), and e(C) = c if both teams or neither have a player within 3 m of the ball. With the ball unknown, e(C) = 0 and e(H) = λ(2p − 1), where p is the team-shape probability that home has the ball and λ = `decoder_shape_weight`. Transitions cost π(H↔A), π(→C) and π(C→); Viterbi finds the best path, exact ties going to C, then H, then A. A span starting at a restart is biased toward the restart team (Laws 8, 13–17). A C run takes the label of the preceding H/A run (else the following), so a loose ball moves a hand-over to where the other team gains control but never creates a spell. In football terms: loose balls belong to the team that had the ball until the other team's control outweighs the switch cost; restarts start with the restart team; ball-unknown live frames keep the current possession unless the team shape says otherwise.

**3.5 Team-shape model** (`features/team_shape.py`, fit in `features/external/fit_team_shape.py`). p(home has the ball) from player shape only, ball hidden. Inputs are home-minus-away differences of eight columns: `adv_x_m`, `adv_vx`, `own_half_frac` (attack-normalised) and `std_x_m`, `std_y_m`, `spread_m`, `mean_speed`, `walking_frac`. The model is p = σ(Σ β_f d_f / s_f), with no intercept and s_f the RMS of d_f, so swapping teams gives exactly 1 − p and no home bias can be learned. It is fitted by Newton/IRLS with L2 = 1.0 fixed a priori on the alive frames of the clean tables of the 5 tuning matches (386,329 rows); there is no search. Five leave-one-match-out (LOMO) fold models are stored so that decoder tuning never scores a match with a model that saw it; the pipeline uses the final model fitted on all 5. Leave-one-match-out accuracy is 0.823 on alive frames and 0.837 on degraded ball-unknown alive frames (J03WN1 weakest at 0.758). The team in possession is wider (std_y), moving toward its attacking goal (adv_vx), and its players run slower on average than the defenders; adv_x and own_half_frac are collinear, so their opposite signs cannot be read separately.

`features/make_submission.py` loads the config through `harness.frozen_config()`, decodes the BAR-ATH table, writes `frame,label` and checks the file with the organisers' validator.

## 4. Protocol

- **IDSSE split, fixed 2026-11-09 before any results:** tune on J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ; hold out J03WPY and J03WR9 for evaluation only. The split is hard-coded (`ef.IDSSE_TUNE`, `ef.IDSSE_HOLDOUT`); `harness.load_table` refuses a held-out table unless `allow_holdout=True` (CLI `--holdout`), and every held-out run is appended to `holdout_log.json`.
- **Held-out policy (user decision, 2026-11-16):** no per-component held-out looks; one final logged evaluation of the integrated pipeline; nothing changed after it.
- **Looks already taken.** The held-out pair is not pristine. Before the final evaluation `holdout_log.json` held three looks: `baseline_nearest` (2026-11-15), `dead_ball_v1` (2026-11-16, detector look 1) and `dead_ball_v1_dsegment_counts` (2026-11-16, detector look 2). The final evaluation is the **fourth** logged look. Label-derived descriptive numbers on the held-out pair were also seen (DFL D share 41–52% and label-run counts over all 7 matches; the held-out columns of the degraded-table ball-loss aggregates). None was used for tuning or selection. Details in Appendix B.4.
- **Harness.** Scoring is the organisers' `pm.evaluate`, not re-implemented, against the DFL proxy label, with 26 `X` rows at the IDSSE period seam. Predictors never see `label` or `ball_dropped`. Differences are judged against the organisers' spellF1 noise band of about 0.04.
- **Tuning rule.** Each tune log records grid, a priori point, fixed keys, seeds, matches, objective and tie rule before the first score. The objective is always mean spell_f1_mean over the 15 degraded tuning runs; ties within 0.002 go to the a priori point (the dead-ball grid first takes the higher alive_dead_f1). When a chosen value sat on a grid edge, the grid was extended once and not again.
- **Enforcement.** `harness.frozen_config()` is the only way the tuned pipeline loads its config. It checks the config against every tracked tune log (chosen rows, `prior_log` chains, and the team-shape model's sha256), and stops on any mismatch. The logs are tracked in `results/tuning/` (aggregate IDSSE scores only, no BAR-ATH data). Details in Appendix B.3.

## 5. Broadcast alignment (built, diagnostic only, not used in the submission)

`features/broadcast_align.py` maps broadcast-tracking frames onto the tactical timeline from tracking alone (no video pixels): constant-offset segments matched on player positions without identities, with replays flagged. Its 7 `broadcast_align_*` config keys are all a priori, since IDSSE has no broadcast feed. The decoder and the submission do not read it. Design: `docs/broadcast-alignment-design.md`.

BAR-ATH label-free aggregates: the coordinate conventions agree (coarse match share 98.6% as given, ≤ 36.5% with teams swapped, y mirrored or rotated); 57.6% of broadcast frames map to live play, 1.4% to replays, 41.0% stay unmapped; player match distance median 0.29 m. Of the 7,720 graded frames called alive whose tactical ball is missing, the broadcast sees the ball on 3,197 (41%).

**Why it is not used (user decision, 2026-11-17).** (1) It cannot be validated on IDSSE, which has no broadcast feed. (2) There are 16–18 distinct time offsets within each half where one live feed should give about one; this is unexplained. (3) Where both see the ball, the broadcast and tactical positions differ by 0.76 m median but 5.4 m at p90, also unexplained. The gain would be at most about 2% of graded frames.

## 6. Results

### 6.1 Tuning split, in-sample (degraded, 15 runs)

| config | spell_f1_mean | macro_f1 | team acc (GT alive) | pred spells / match (true 454) |
|---|---|---|---|---|
| dead_ball_v1 + nearest team | 0.121 | 0.787 | 0.722 | 1715 |
| decoder, run 2 | 0.391 | 0.830 | 0.785 | 275 |
| decoder + team shape, run 3 (submitted) | 0.418 | 0.842 | 0.804 | 300 |

- The detector lifts alive_dead_f1 to 0.95 but spellF1 barely moves, because the nearest-player call flips during passes and duels; the decoder fixes that fragmentation and now slightly under-segments.
- **Team-shape weight.** `decoder_shape_weight` is 2.0, the top edge of its run-3 grid {0, 0.125, 0.25, 0.5, 1.0, 2.0}. Gains are flattening, about +0.003 spell_f1 per doubling (1.0 → 0.4151, 2.0 → 0.4178). It was not extended (user decision, 2026-11-17).

### 6.2 BAR-ATH, label-free plausibility (no ground truth, no video)

| submitted config (run 3) | H | A | D | X (ungraded) | sha256 |
|---|---|---|---|---|---|
| share of 158,857 graded frames | 53,421 (33.6%) | 37,002 (23.3%) | 68,434 (43.1%) | 40,462 | 17511abf… |

- Predicted D share 42.0% excluding the published 2nd-half pre-kickoff span, inside the IDSSE range of predicted D shares (40.6–44.5%). 145 D segments (144 excluding that span), more than the IDSSE true mean of 104.8 per match, as the detector also over-counts on IDSSE.
- 88.1% of ball-missing graded frames are called D; the team-shape model acts only on the remaining 11.9%.

### 6.3 Reproducibility

`sh reproduce.sh` in a fresh clone with the raw data unpacked into `data/` rebuilt the submission byte for byte (tested at commits 80a2541, 60b2645 and 8e8c215, the final pipeline commit; later commits change only docs; sha256 17511abf…; Appendix C).

### 6.4 Final held-out evaluation

The one final held-out evaluation of the full pipeline (`harness.py --predictor decoder_v1 --holdout --tables degraded --seeds 0 1 2`), run 2026-11-17T11:10:06 at commit b4cc6ed with the submitted config (sha256 ef6fb3b4…, the same file `make_submission.py` uses). It is the fourth logged look at J03WPY/J03WR9; nothing in the pipeline or config was changed after it. The log entry is the last one in `results/holdout/holdout_log.json`, a tracked copy of `holdout_log.json` (IDSSE aggregates only).

| 6 runs (2 matches × 3 seeds) | spell_f1_mean | spellF1 @ 0.5 | macro_f1 | alive_dead_f1 | team acc (GT alive) | pred spells / match (true) |
|---|---|---|---|---|---|---|
| held out, mean | **0.417** | 0.424 | 0.851 | 0.950 | 0.824 | 279 (396) |
| held out, min–max | 0.399–0.434 | 0.401–0.450 | 0.848–0.856 | 0.947–0.953 | 0.819–0.828 | 269–287 (370–421) |
| in-sample, run 3 (tuning matches, for comparison) | 0.418 | | 0.842 | | 0.804 | 300 (454) |

- Held out matches in-sample (0.417 vs 0.418), so the tuning did not overfit the five tuning matches in any way this pair can show. Two matches and three ball-loss seeds are a small sample: the per-run range is 0.40–0.43.
- The decoder still under-segments (279 predicted vs 396 true spells per match). The held-out matches have fewer true spells than the tuning matches (396 vs 454), so spell counts are not directly comparable across the two rows.
- alive_dead_f1 is identical to detector look 1 (Appendix B.4), as expected: the decoder keeps the detector's D frames fixed and the tables and seeds are the same, so only the H/A calls changed.
- This measures the pipeline on IDSSE with simulated ball loss. It does not measure the transfer to BAR-ATH (Section 7).

## 7. Limitations and risks

- **Transfer from clean IDSSE player tracking.** Only the ball is degraded; IDSSE players stay clean. Several tuned values sit where clean and noisy tracking differ most: dead-ball smoothing 0 and lag 0, walking 0.7, and decoder margin 0.25 m (close to the sign of a noisy margin, while the config assumes about 0.5 m ball error). Treat them as weakly supported on BAR-ATH.
- **Grid edges.** Detector lag and smoothing (both at a physical minimum); all three run-2 decoder keys (c_exit 25 is not a physical bound); the team-shape weight 2.0. The one-extension rule or a user decision stopped further search.
- **Unidentified parameters.** At the chosen entry/exit costs a hand-over through C costs at most about 28, less than the switch penalty of 50, so switch_penalty and restart_bias have no effect; their values rest on their a priori reasons.
- **Flat dead-ball surface.** spell_f1 does not identify the detector keys; between runs 1 and 2, lag went 12 → 0 and min_alive 0 → 25.
- **The ball-loss model is an assumption** (drop probabilities, lag spreads, gap rates), not identifiable from allowed data, and not blind to BAR-ATH aggregates. It does not match BAR-ATH exactly (ball seen 48.1% vs 59.3%; long-gap median 15.1 s vs 11.6 s; too few very long gaps).
- **Proxy labels.** DFL's alive/dead timing and possession hand-over may follow a different convention from the organisers' truth; all boundary placement learned on IDSSE inherits it.
- **Segmentation.** D is over-segmented (14–37% too many segments on IDSSE; 144 on BAR-ATH vs 104.8); possessions are under-segmented (300 vs 454 per match).
- **Restart team.** Smart Tagging tag meanings are assumptions; goal kick vs corner is not decided by geometry; throw-ins fall back to the nearest player.
- **Team-shape transfer.** The model learned Bundesliga shapes from clean tracking; BAR-ATH has noisier tracking and a possession style (long, wide home possessions) that may change the shape-to-possession map. Antisymmetry removes home bias, not team-style bias. At weight 2.0 a confident shape frame counts as two clear ball frames, so the a priori rationale for 0.5 (about 4–5 s of shape evidence to switch inside a gap) no longer holds (our reading of the score scale, not a measurement).
- **Held-out pair.** Not pristine: four logged looks, three before the final evaluation, plus label-derived descriptive numbers (section 4).

## 8. Credits and licences

- **IDSSE data.** Bassek, Rein, Weber et al. "An integrated dataset of spatiotemporal and event data in elite soccer." *Scientific Data* 12, 195 (2025). doi:10.1038/s41597-025-04505-y. Licence CC BY 4.0. Downloaded from huggingface.co/datasets/pysport/idsse-data. We used the 7 matches J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ, J03WPY and J03WR9, converted to compact arrays; the raw XML was not kept.
- **kloppy** (3.19.1, BSD-3-Clause): IDSSE loading, and the URL pattern and orientation check in `fetch_idsse.py` / `idsse_features.py`.
- **The organisers' evaluation code** (`evaluation/possession_metrics.py`, `validate_submission.py`; MIT licence, copyright Metrica Sports): the only scorer used, on IDSSE (`pm.evaluate`) and to validate the submission.
- **Provided match data** (BAR-ATH tactical ATD and Smart Tagging): used under the competition's terms, locally only, not redistributed.
- Published references for parameters: IFAB Laws of the Game (Laws 2, 8, 9, 13–17); Bradley et al. 2009 (speed zones); Bohannon 1997 (walking speed). The full list is in `features/config.json` `_provenance`.

---

## Appendix A. Parameters

All 44 pipeline keys in `features/config.json` are listed below (keys starting with `_` excluded). The config also holds 7 `broadcast_align_*` keys (a priori, diagnostic only, not read by the decoder or the submission; see section 5 and `docs/broadcast-alignment-design.md`). The full source and reason for each is in `_provenance`. For tuned keys, the tuning data are the IDSSE tuning matches J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ: degraded tables (a priori ball loss), seeds 0–2, 15 runs. The target is the DFL proxy labels via `pm.evaluate`, and the objective is mean spell_f1_mean.

### A.1 Feature extraction

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

Extraction details: ball velocity is a central difference over ±2 frames, masked at block edges; stillness is the share of frames within ±12 whose speed is below 0.5 m/s; player velocity comes from positions, not the raw speed field (missing on about 1.63 tracked slots per frame); walking means speed < 2.0 m/s; near-ball counts use 3 m and 10 m; on BAR-ATH only 51 frames are filled by interpolation. Kickoff centroids are averaged over a 1 s window and must be at least 5 m apart along x. `nearest_team` is 0 = H, 1 = A, −1 = unknown; `nearest_margin_m` is positive when home is closer. Smart Tagging gives one boolean per (code, team) per frame.

### A.2 Ball-loss simulation (IDSSE only; never applied to BAR-ATH)

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

Simulation details: each match is degraded per period with its own RNG `default_rng([seed, match_index])`. Each DFL dead span [s, e) maps to a ball gap [s + L_on, e + L_re), clipped to the period, with L_on ~ U{0..100} and L_re ~ U{0..25} frames drawn per span; a gap of at least 126 frames is dropped with probability 1.0, a shorter one with 0.5. Live gaps are a Poisson process with lengths log-uniform on [126, 500] frames; flickers are uniform on [1, 12] frames. The lags are random so that ball vanish/reappear edges do not sit at a fixed offset from DFL's dead-ball edges. The degraded tuning tables show 157.7 runs ≥ 5 s per BAR-ATH-length match (scaled to 158,857 graded frames), against 132 on BAR-ATH.

### A.3 Dead-ball detector

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

Detector details. Let W_t be the walking share (the minimum of the two teams' shares, or `all_walking_frac` where a team's share is NaN) and V_t the all-player mean speed, both smoothed by a NaN-aware centred mean over ±h frames inside the block (h = `dead_smooth_half_window_frames`), giving W̄_t and V̄_t; stopped_t = [W̄_t ≥ w_min] ∧ [V̄_t ≤ v_max], NaN counting as not stopped. Candidates: (b) out of play, `ball_out_m` > 0.61 m; (c) placed ball, `ball_still_frac` ≥ 0.5 and stopped_t; (a) long gap, ball unknown for at least `dead_min_gap_frames` and not live (a gap is live when the median of W̄ over it is below w_min or the median of V̄ above v_max; a gap with no finite player data is not live); (d) known dead, frames [110808, 113742) on BAR-ATH. For a long gap [v, e): end = e + `dead_reappear_offset_frames`; start = v if the ball was last seen out of play, else onset − `dead_decel_lag_frames` where onset is the start of the stopped run containing v (if that run reaches back more than 125 frames, start = v − 125 with no lag), else the first stopped frame inside the gap minus the lag, else v + `dead_vanish_offset_frames`. Alive runs shorter than `dead_min_alive_frames` between two D candidates in one block become D; D segments shorter than `dead_min_dead_frames` (0, off) become alive. Restart team priority: (1) a single `SET PIECES` tag team (H/A) overlapping the segment or its restart frame (`N/A` falls through); (2) a `GOALS` tag inside the segment, conceding team kicks off; (3) the nearest player's team at the restart frame; (4) unknown. The goal-line rule (goal kick vs corner) is deferred to v2, because it needs a last-touch proxy.

### A.4 Possession decoder

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

Decoder details. The full objective is ẑ = argmax_z [ b(z_0) + Σ_t e_t(z_t) − Σ_{t≥1} π(z_{t−1}, z_t) ]; staying costs 0. At λ = 0 all three ball-unknown scores are 0, which is decoder v1, and the transition costs alone carry the state through the gap. Restart bias: if the span starts at the end of a detector D segment whose restart team is H or A, b = 0 for that team and −`decoder_restart_bias` for the other two states, else b ≡ 0; a span at the start of a block gets no bias. C resolution: a span that is C throughout takes the majority `nearest_team` call over its ball-known frames; if none or tied, the last H/A label before the span in the block, else the next one after it, else D.

Team-shape coefficients (final model, scaled units): adv_x −1.95, adv_vx +0.71, own_half_frac −1.29, std_x +0.23, std_y +1.51, spread +0.34, mean_speed −0.61, walking_frac −0.03. The fit minimises −Σ[y log p + (1 − y) log(1 − p)] + (L2/2)·|β|² with y = 1 for H; L2 = 1.0 is a unit Gaussian prior on each scaled coefficient, which barely shrinks at about 3·10⁵ frames. The eight h_/a_ column pairs are checked to be bit-identical in all 15 degraded tables, so degradation touches only the ball; `test_team_shape.py` asserts that no ball-derived column is used. Model, fold models, counts, metrics and data sha256 are in `results/tuning/team_shape/team_shape_model.json`.

## Appendix B. Tuning and evaluation detail

### B.1 Harness and tie rules
- **Scoring.** `evaluation/possession_metrics.py` (`pm.evaluate`). The two IDSSE periods sit in adjacent rows, so 26 `X` rows (one more than the widest transition tolerance, 25) are inserted at the seam in truth and prediction, keeping every spell and transition match within one half.
- **Leak guard.** Predictors get the table with `label` and `ball_dropped` removed, and the detector raises an error if it sees either.
- **Runs.** Degraded tables give one run per (match, seed), 15 on the tuning split; clean tables one per match. Only `ball_loss_variant = a_priori` tables load. Noise band about 0.04 (`SPELL_NOISE_BAND`, EVALUATION.md).
- **Tie rules.** Dead-ball joint grid: among combos within 0.002 of the best, the highest mean alive_dead_f1, then the smallest grid-index distance to the a priori point, then grid order. `dead_min_dead_frames`: if the top values are within 0.001, the smallest value. Decoder: among combos within 0.002 of the best, the smallest grid-index distance to the a priori point, then grid order.

### B.2 Runs

| run | log | keys | combos | chosen |
|---|---|---|---|---|
| dead-ball run 1 (2026-11-15T23:58) | `tune_log_run1.json` | six player keys | 10,000 | lag 12, smoothing 6 (edge), walking 0.7, speed 2.0, min_gap 50, min_alive 0 (edge) |
| dead-ball run 2, edge extension (2026-11-16T01:29) | `tune_log.json` | same; smoothing grid extended once to {0, 3} | 15,000 | lag 0 (edge), smoothing 0 (edge), walking 0.7, speed 2.0, min_gap 50, min_alive 25 |
| `dead_min_dead_frames` (2026-11-16T10:21) | `tune_log_min_dead.json` | one key over {0, 12, 25, 38, 50} | 5 | 0 |
| decoder run 1 (2026-11-16T10:33) | `decoder_tune/tune_log.json` | six decoder keys, 3 values each | 729 | margin 1.0, c_score 0.5, switch 50, entry 12, exit 12, restart_bias 50 |
| decoder run 2, edge extension (2026-11-16T23:03) | `decoder_tune/tune_log_run2.json` | margin {0.25, 0.5, 1.0, 2.0}, entry and exit {3, 6, 12, 25} | 64 | margin 0.25, entry 3, exit 25 |
| team-shape fit (2026-11-16T23:22) | `team_shape/team_shape_model.json` | logistic coefficients, by log loss; no grid | — | final model + 5 LOMO fold models |
| decoder run 3, team-shape weight (2026-11-16T23:23) | `decoder_tune/tune_log_run3.json` | `decoder_shape_weight` over {0, 0.125, 0.25, 0.5, 1.0, 2.0}, all other keys at run-2 values | 6 | 2.0 (top edge) |

**Run 3.** The a priori value 0.5 and the grid (`GRID_RUN3`) were declared in `tune_decoder.py` before the team-shape fit and before the run. Run 3 uses the same tables, seeds, objective and tie rule as runs 1–2. It records the run-2 log as `prior_log`, and the model file's path and sha256 as `shape_model`. Each tuning match gets p_home from its LOMO fold model (the fold must not have seen that match; the guard raises an error otherwise). The weight 2.0 is on the top edge; it was not extended (user decision, 2026-11-17).

**Grid-edge rule: one extension only.**
- **Dead-ball run 2.** The only extension was the smoothing grid. min_alive could not be extended, because 0 is its hard lower bound. Lag 0 and smoothing 0 are both edges and physical minimums.
- **Decoder run 2.** All three chosen values are edges of the run-2 grid. 0.25 m and 3 are near the physical floor of 0. Exit 25 is not a physical bound, and exit 50 was never tried together with entry 3 or 6. Run 2 recorded the run-1 log as `prior_log`. The run-1 point reproduced exactly in run 2 (0.38136).
- **Why run 2 is kept over run 1.** It rests on the declared procedure, not on evidence of a better optimum: the gain is +0.009, inside any reasonable noise band.

### B.3 Enforcement and code hashes
- `harness.frozen_config()` is the only way the tuned pipeline loads its config (harness predictors, `make_submission.py`). It runs `check_provenance`, then checks:
  - every tuned key names a tune log that `frozen_config` actually checks;
  - the config equals the chosen rows of both dead-ball logs;
  - the decoder keys not re-tuned in run 2 equal the run-1 log;
  - the run-2 keys equal the run-2 log;
  - the run-2 log's `prior_log` sha256 equals the run-1 log;
  - `decoder_shape_weight` equals the run-3 log, and the run-3 log's `prior_log` is the run-2 log;
  - `decoder_shape_model` is the tracked model file, and its sha256 equals the one run 3 was tuned with.
- `tune_dead_ball.py --check --log <log>` prints the detector check on its own. There is no separate `--check` for the decoder: the decoder logs are checked only through `harness.frozen_config()`, which runs on every `make_submission.py` call and every harness `decoder_v1` run.
- The tune logs were moved to `results/tuning/` from `external_data/` unchanged (same sha256). Their internal `previous_run` / `prior_log` / `tune_log` fields still name the old `external_data/` paths.
- The logs also record the sha256 of the script, harness, detector and decoder, and run 3 also records the team-shape module. Checked against the files:
  - `dead_ball.py` matches every log.
  - `harness.py` and `team_shape.py` match the run-3 log and the model file. `possession_decoder.py` does not: its docstring was corrected after run 3 (text only; the C-score wording for ball-known, margin-NaN frames).
  - **2026-11-17:** the fold-leakage guard in `features/external/tune_decoder.py` `_init` and the data-identity guards in `features/external/fit_team_shape.py` `load_tune` were changed from `assert` to `raise ValueError` (behaviour unchanged when the checks pass). So `tune_decoder.py` no longer matches the run-3 log, and `fit_team_shape.py` no longer matches the team-shape model record.
  - The hashes logged for `harness.py` and `tune_decoder.py` in the dead-ball logs and decoder runs 1–2 no longer match, and `tune_dead_ball.py` matches none of its logs. Later edits explain this: the `frozen_config` checks for runs 2 and 3, docstring log paths updated when the logs moved, the `--run3` option, and p_home passed to the decoder.
  - The search code of runs 1–2 is unchanged (findings). At weight 0 the run-3 code reproduces run 2's 0.39054 exactly.
- Open review follow-ups (not done, user decision 2026-11-17): a LOMO diagnostic refit without `own_half_frac`/`adv_x` and without `spread`; tests for `team_shape.load()`, `decode()` end to end, the per-match fold selection in `tune_decoder._init`, and the ball-known/margin-NaN case; pinning `team_shape.py`'s sha in the model check; for broadcast alignment, the top_k plateau `<`, replay rewind depth, counting on feature-table unfilled frames, orientation split by half plus extra variants, progress printing and more tests.

### B.4 Detailed results

**Tuning split, in-sample, detector stage.** Mean (min–max) over runs.

| predictor | tables | runs | spell_f1_mean | spell_f1@0.5 | macro_f1 | alive_dead_f1 | team_acc_gt_alive | n_pred_spells | fragmentation |
|---|---|---|---|---|---|---|---|---|---|
| baseline_nearest | degraded | 15 | 0.106 (0.08–0.13) | 0.092 (0.07–0.11) | 0.720 (0.69–0.75) | 0.861 (0.81–0.90) | 0.617 (0.55–0.67) | 1765 (1529–1994) | 3.45 (2.78–4.25) |
| dead_ball_v1 untuned | degraded | 15 | 0.120 (0.09–0.14) | 0.106 (0.08–0.13) | 0.756 (0.73–0.80) | 0.917 (0.90–0.95) | 0.730 (0.70–0.75) | 1704 (1471–1953) | 3.26 (2.60–4.11) |
| dead_ball_v1 tuned | degraded | 15 | 0.121 (0.09–0.14) | 0.107 (0.08–0.13) | 0.787 (0.77–0.81) | 0.949 (0.93–0.97) | 0.722 (0.69–0.75) | 1715 (1486–1974) | 3.31 (2.65–4.20) |
| baseline_nearest | clean | 5 | 0.100 (0.08–0.12) | 0.086 (0.07–0.10) | 0.359 (0.33–0.38) | 0.704 (0.65–0.74) | 0.764 (0.75–0.78) | 2092 (1981–2240) | 3.77 (3.32–4.43) |
| dead_ball_v1 untuned | clean | 5 | 0.115 (0.09–0.14) | 0.101 (0.08–0.12) | 0.776 (0.76–0.81) | 0.933 (0.92–0.96) | 0.757 (0.74–0.77) | 2124 (2014–2271) | 4.09 (3.55–4.84) |
| dead_ball_v1 tuned | clean | 5 | 0.117 (0.09–0.14) | 0.103 (0.08–0.13) | 0.804 (0.78–0.83) | 0.961 (0.95–0.98) | 0.754 (0.74–0.77) | 2108 (1988–2264) | 4.06 (3.52–4.85) |

The tuning matches have 454 (401–511) true spells per match. "Nearest team" fills the alive frames: the nearest player's team, carried forward over unknown-ball frames. On clean tables the ball is never missing, yet alive_dead_f1 is 0.96, so the player cues do the work there. The predictions are 3–4 times more fragmented than the truth.
- **Flat surface.** In dead-ball run 2, mean spell_f1 spans only 0.111–0.122 across all 15,000 combos, and 1,768 combos are within the 0.002 tie band; the choice was in practice made by alive_dead_f1. Chosen combo: spell_f1 0.1212, alive_dead_f1 0.9494. Only the walking and speed thresholds move alive_dead_f1 much.
- **Clean cross-check** (`crosscheck.json`). Three keys are flagged: smoothing, min_alive and walking. Only walking matters in practice: clean alive_dead_f1 is 0.961 at 0.7, against 0.973 at 0.6, the grid edge. The chosen values were not changed.

**Tuning split, in-sample, decoder stage (degraded, 15 runs).**

| config | spell_f1_mean | macro_f1 | team acc (GT alive) | pred spells / match | true spells / match |
|---|---|---|---|---|---|
| dead_ball_v1 + nearest team | 0.121 | 0.787 | 0.722 | 1715 | 454 |
| decoder, a priori values | 0.277 | — | — | 181 | 454 |
| decoder, run 1 | 0.381 | 0.827 | 0.781 | 289 | 454 |
| decoder, run 2 | 0.391 | 0.830 | 0.785 | 275 | 454 |
| decoder + team shape, run 3 (current) | 0.418 | 0.842 | 0.804 | 300 | 454 |

Best in-sample spell_f1 per value of each key, other keys free:
- **Run 1.** margin 1.0 / 2.0 / 4.0: .381 / .365 / .349; c_score 0 / 0.5 / 1.0: .372 / .381 / .345; switch: .381 for every value; entry, and likewise exit, 12 / 25 / 50: .381 / .372 / .363; restart_bias 0: .375, 50 and 100: .381.
- **Run 2.** margin 0.25 / 0.5 / 1.0 / 2.0: .391 / .388 / .382 / .372; entry 3 / 6 / 12 / 25: .391 / .385 / .388 / .386; exit 3 / 6 / 12 / 25: .386 / .382 / .388 / .391.

The good combos lie on a ridge where entry + exit ≈ 24–28: the surface pins the total cost of going through C, not how it is split.

**Team-shape model, leave-one-match-out** (each model fitted on 4 tuning matches, predicting the 5th; accuracy with p > 0.5 → H / mean log loss; held-out matches not used).

| match | alive frames, clean | ball-unknown alive frames, degraded (3 seeds) |
|---|---|---|
| J03WMX | 0.841 / 0.377 | 0.856 / 0.350 |
| J03WN1 | 0.758 / 0.511 | 0.756 / 0.518 |
| J03WOH | 0.838 / 0.372 | 0.854 / 0.341 |
| J03WOY | 0.826 / 0.393 | 0.863 / 0.335 |
| J03WQQ | 0.841 / 0.350 | 0.852 / 0.347 |
| pooled | 0.823 / 0.397 (n 386,329) | 0.837 / 0.377 (n 217,375) |

**Run-3 profile** (degraded, 15 runs, in-sample; p_home from the LOMO fold models):

| decoder_shape_weight | 0 | 0.125 | 0.25 | 0.5 (a priori) | 1.0 | 2.0 (chosen) |
|---|---|---|---|---|---|---|
| spell_f1_mean | 0.3905 | 0.3990 | 0.4042 | 0.4113 | 0.4151 | **0.4178** |
| team acc (GT alive) | 0.785 | 0.796 | 0.800 | 0.802 | 0.803 | 0.804 |
| pred spells / match | 275 | 278 | 283 | 289 | 294 | 300 |

- Weight 0 reproduces run 2 exactly (0.3905). 1.0 is 0.0026 below 2.0, just outside the 0.002 tie band; the curve is flattening (+0.004, then +0.003 per doubling).
- The gain from the a priori 0.5 to 2.0 is 0.006, small next to the per-run spread (0.39–0.47). The gain from run 2 is +0.027 in-sample.
- **Fold models vs final model.** An ad-hoc harness `decoder_v1` run on the tuning tables, using the final all-5 model in place of the fold models, also gave 0.418 (not logged, not used for any choice).

**Earlier held-out looks (J03WPY, J03WR9; evaluation only; nothing tuned or selected on them).**

| predictor | tables | spell_f1_mean | macro_f1 | alive_dead_f1 | team_acc_gt_alive | n_pred_spells |
|---|---|---|---|---|---|---|
| baseline_nearest (fixed-lag degraded tables, not re-run) | degraded (6 runs) | 0.103 | 0.757 | 0.896 | 0.681 | 1792 |
| baseline_nearest | clean (2 runs) | 0.095 | 0.376 | 0.725 | 0.797 | 1991 |
| dead_ball_v1 tuned (look 1) | degraded (6 runs) | 0.115 | 0.797 | 0.950 | 0.746 | 1681.5 |
| dead_ball_v1 tuned (look 1) | clean (2 runs) | 0.113 | 0.820 | 0.965 | 0.784 | 2012 |

- Log entries: `baseline_nearest`, degraded and clean (2026-11-15T20:40:30); `dead_ball_v1`, degraded and clean (2026-11-16T03:06:19); `dead_ball_v1_dsegment_counts` (2026-11-16T03:06:59), logged by hand, whose `script_sha256` is the harness's, not the script's.
- The held-out pair has 396 (370–421) true spells per match. The 2012 for dead_ball_v1 clean is from `holdout_log.json`; findings lists it as n/a.
- alive_dead_f1 on held-out degraded tables is 0.950, against 0.949 in-sample. Held-out spell_f1 (0.115) lies inside the flat in-sample band, so it says nothing about overfitting.
- **Look 2, D-segment counts** (`dead_segment_counts.py`): 126.5 predicted vs 94.5 true D segments per match on degraded tables, 147.5 vs 94.5 on clean. Predicted segments shorter than 1 s: 16.8% (degraded), 18.8% (clean). Predicted D share 42.6% vs 43.1% true (degraded). The tuning matches show the same pattern: 119.5 and 143.6 vs 104.8.
- After look 2 the held-out pair is no longer clean for judging any change motivated by D-segment counts. `dead_min_dead_frames` was proposed after that look; it was tuned on the tuning matches only and came out at 0, which leaves the detector identical to the version scored on held-out.
- Label-derived descriptive numbers also seen, not used for selection: the DFL D share (41–52%) and label-run counts (462–614) over all 7 matches in the adapter check; the held-out columns of the degraded-table ball-loss aggregates (ball seen share, gap counts and lengths) for both simulator variants.

**BAR-ATH, label-free detail.**
- Dead-ball detector (`features/dead_ball_report.py`): D share 43.1% of graded frames (42.0% excluding the pre-kickoff span), against IDSSE tuning predicted D shares of 42.4% (clean) to 44.5% (degraded), split means. 145 D segments (65 in the 1st half, 80 in the 2nd); IDSSE true range 99–113. 13 segments (9.0%) are shorter than 1 s. Length quartiles 2.7 / 9.8 / 21.6 s; longest 221.2 s. Restart team: A 76, H 65, unknown 4. 78 of 145 segments overlap a set-piece tag (an upper bound). 16.8% of D frames have the ball seen.
- Submission history (shares of graded frames; run 1 converted from findings' all-frame shares, ±0.1 pp rounding):

| config | H | A | D | X (ungraded frames) | sha256 |
|---|---|---|---|---|---|
| decoder run 1 (history) | 33.2% | 23.7% | 43.0% | 40,462 | not recorded |
| decoder run 2 (history) | 51,439 (32.4%) | 38,984 (24.5%) | 68,434 (43.1%) | 40,462 | d47ab2db… |
| **decoder run 3, team shape (submitted)** | **53,421 (33.6%)** | **37,002 (23.3%)** | **68,434 (43.1%)** | 40,462 | **17511abf…** |

- Run 1: all-frame shares H 26.5%, A 18.9%, D 34.3%, X 20.3%; the validator counted 426 possessions with a mean of 8.5 s (the a priori decoder gave 224 of 16.1 s). Run 2: reproduced byte for byte from a fresh clone at 80a2541. Run 3: D unchanged from run 2 (the model only re-scores alive frames); 1,982 graded frames moved from A to H on net; the D-segment report is unchanged (145).

**Broadcast alignment, further detail** (`features/broadcast_align.py --report`). Broadcast ATD 164,488 frames, 51 slots; orientation shares 98.6% as given vs 31.2% teams swapped, 36.5% y mirrored, 36.5% rotated 180°. 197 live segments, 26 replay segments, 7 dropped; live map strictly increasing. Player match distance median 0.29 m, p90 0.53 m. 37 distinct offsets overall, distinct live offsets per tactical block {0: 5, 1: 16, 3: 18}. Of 64,643 graded frames with the raw tactical ball missing, alive 7,720 (3,197 with broadcast ball) and D 56,923 (2,674 with broadcast ball). Ball consistency over 69,241 frames where both see it: median 0.76 m, p90 5.4 m. The counts use the raw tactical ball, so they include short gaps the feature table already interpolates.

## Appendix C. How to rerun

### C.1 BAR-ATH submission
From a fresh clone, with the provided data unpacked into `data/`:

    sh reproduce.sh

It was developed and tested with Python 3.14.0. `reproduce.sh` calls `python3` and does not pin the interpreter version. The script:
1. creates `.venv` and installs the pinned `features/requirements.txt` (numpy 2.5.3, pandas 3.0.5, pyarrow 26.0.0);
2. runs `features/extract_features.py`, which writes `data/derived/features_tactical.parquet`;
3. runs `features/make_submission.py`, which loads the config through `frozen_config()` (checked against the tracked tune logs and their sha256), decodes, writes `data/derived/submission.csv`, validates it with the official validator, and prints `wrote … (199319 rows, valid)` and the H/A/D share of graded frames plus the X count;
4. prints `shasum -a 256 data/derived/submission.csv`. For the run-3 config: 17511abfaffd9e4f57404f20d24c19b7b1145633b8c050ed052213fdcff52053.

It needs no IDSSE data, no tuning and no network access beyond pip. It stops with an error if the config does not match the logs. A label-free sanity summary is available from `python features/extract_features.py --report`.

### C.2 IDSSE pipeline (external data, optional)
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
- The new logs record current script hashes, which differ from the logged ones (Appendix B.3). Compare the `chosen` and `results` fields, not the file sha256.
- Held-out runs need `--holdout` and are logged. Under the policy in section 4, only the final evaluation may use it.

### C.3 Tests
Synthetic-match tests, run after any change to the extractor (the command from CLAUDE.md):

    python features/tests/make_synthetic_match.py /tmp/synth && MTAH_DATA_DIR=/tmp/synth python features/tests/test_synthetic.py && python features/tests/test_fetch_checks.py && python features/tests/test_idsse_features.py && python features/tests/test_degrade_ball.py && python features/tests/test_harness.py && python features/tests/test_dead_ball.py && python features/tests/test_tune_dead_ball.py && MTAH_DATA_DIR=/tmp/synth python features/tests/test_possession_decoder.py && python features/tests/test_team_shape.py && python features/tests/test_broadcast_align.py
