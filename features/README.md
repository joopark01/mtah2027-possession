# Per-frame feature table

```bash
pip install -r features/requirements.txt
python features/extract_features.py --report
```

Reads the tactical ATD, its metadata and the Smart Tagging from `data/` (or `MTAH_DATA_DIR`). It writes three files to `data/derived/`, which is gitignored:

- `features_tactical.parquet`: 199,319 rows, one per video frame (= submission row), about 60 columns
- `features_tactical_provenance.json`: the config used, input file fingerprints, script hash, git commit and package versions
- `atd_tactical_cache.npz`: the parsed raw tracking, so re-runs skip parsing. It is reused only while the raw file's size and mtime are unchanged.

`--report` prints a label-free sanity summary over the graded frames: ball visibility, tracked objects, speed ranges, out-of-pitch share, and any extra raw ball fields. Where DATA.md publishes a figure, it is shown alongside. **The report reads no ground truth.** Under the organisers' 2026-11-03 rules, nothing in this pipeline may be fitted or calibrated on it.

**Never commit or upload the outputs.** They are frame-level derived data (RULES.md).

## Columns

| group | columns |
|---|---|
| meta | `frame`, `half` (0 = ungraded), `graded` |
| ball | `ball_seen`, `ball_interp`, `ball_x_m`, `ball_y_m`, `ball_gap_len`, `ball_since_seen`, `ball_until_seen`, `ball_vx`, `ball_vy`, `ball_speed`, `ball_accel`, `ball_turn_deg`, `ball_still_frac`, `ball_out_m` (> 0 outside the pitch), `ball_raw_f2…` (undocumented raw fields, kept as-is), `ball_dist_centre_m` |
| team (`h_` / `a_`) | `n_tracked`, `cx_m`, `cy_m`, `std_x_m`, `std_y_m`, `spread_m`, `mean_speed`, `walking_frac`, `adv_x_m`, `adv_vx` (along the team's attacking direction), `own_half_frac`, `dmin_ball_m`, `d2_ball_m`, `n_within_{3,10}m` |
| cross | `nearest_team` (0 H, 1 A, -1 none), `nearest_margin_m` (> 0 = home closer), `nearest_free_space_m`, `all_mean_speed`, `all_walking_frac`, `n_tracked`, `n_tracked_no_speed` (raw speed field NaN while the position is present), `n_implausible_velocity` |
| Smart Tagging | `st_<code>_<H/A/NA>`: true while that tag is active |

## Conventions

- **Units:** metres and m/s. Normalised x runs along the pitch length (105.3 m) and y along the width (68.0 m). The orientation is checked at both kickoffs, and the script stops if it doesn't hold.
- **Attack direction:** read off the kickoff formation, and checked to flip at halftime.
- **Velocities:** central difference over ±`velocity_half_window_frames`.
- **Player speed and walking:** `mean_speed`, `walking_frac` and `adv_vx` come from position velocity (jump-filtered, see below), not the raw speed field, and are NaN where the ±k window leaves the half.
- **Player identity switches:** a per-track velocity above `player_max_plausible_speed_m_s` is set to NaN and counted in `n_implausible_velocity`.
- **Ball gaps:** a gap is filled linearly only if it is at most `ball_max_interp_gap_frames` long and the ball is seen on both sides. Nothing is filled across a half boundary.
- **Thresholds:** every one is in `config.json`; there are no defaults in the code.

## Parameter provenance

Every value in `config.json` has an entry in its `_provenance` block with `basis`, `source`, `reason` and `tuned_on`. The script refuses to run if any value lacks one. Allowed bases:

| basis | meaning | today |
|---|---|---|
| `published` | stated in the challenge docs or a standard reference | fps, kickoff frames, walking speed, team map |
| `law_of_game` | follows from the IFAB Laws of the Game | kickoff window and separation, still-ball speed, near-ball radii |
| `physics` | follows from physics or human limits | turn-angle min speed, max plausible player speed, `dead_out_margin_m` |
| `a_priori` | design choice made before seeing any labels | ball gap limit, velocity window, still-ball window, all `ball_loss_*` (IDSSE ball-loss simulation), the other `dead_*` (dead-ball detector) |
| `tuned_external` | tuned on an external dataset, named in `tuned_on` | none active (the unused fitted ball-loss variant is in `features/external/ball_loss_fitted_bar_ath.json`) |

**Rules for changing a value:**
- **Tuning a value:** set `basis` to `tuned_external` and name the external dataset in `tuned_on`. The check rejects anything naming the provided game (`BAR-ATH`), because the organisers' 2026-11-03 rules forbid tuning on its ground truth or video.
- **Revisiting a value:** the optional `revisit` field lists values planned for external tuning.

The full block is copied into the provenance JSON written next to each output. Use it as the parameter table for the tech doc.

## IDSSE features

`features/external/idsse_features.py --all` (or a list of ids) turns each `external_data/idsse/<id>.npz` into `external_data/idsse_features/<id>.parquet`, plus `<id>_provenance.json`. It calls `extract_features.features_from_arrays`, so the features come from the same code as BAR-ATH. The data comes from Bassek, Rein, Weber et al., Sci Data 12, 195 (2025), CC BY 4.0. Differences from the BAR-ATH table:

- There are no `st_` columns, because IDSSE has no Smart Tagging.
- Every frame is graded.
- `ball_raw_f2` is ball z in metres.
- The extra `label` column holds H/A/D, from DFL BallStatus and BallPossession.

This step only adapts the data. It does no degradation and no tuning.

`features/external/degrade_ball.py --all --seeds 0 1 2` simulates BAR-ATH-style ball loss on top of it and writes `external_data/idsse_degraded/<id>_s<seed>.parquet`. It uses the a priori `ball_loss_*` values from `config.json`. `--ball-loss-params features/external/ball_loss_fitted_bar_ath.json` swaps in the unused variant fitted to BAR-ATH label-free aggregates, which needs `tune_log.json`. See docs/findings.md.

## Dead-ball detector (`dead_ball.py`)

Rule-based v1 of `docs/dead-ball-detector-design.md`. `detect(df, cfg, ctx)` returns `dead` (bool per row), `segments` (half-open `[start, stop)`, never crossing a `half` block, so the adjacent IDSSE periods stay apart) and `restart_team` per segment (H, A or unknown). `predict_labels(df, cfg, ctx)` gives D on dead frames. Alive frames get the nearest player's team where the ball position is known; otherwise the last alive team is carried forward within the block, else the next one, else D. That is only there so the harness can score it; the possession decoder replaces it.

- **ctx** is per game: `known_dead` (BAR-ATH `[(110808, 113742)]`, IDSSE `[]`) and `smart_tagging` (whether `st_` columns exist; asserted against the table). Pitch and attack signs are not needed yet: `ball_out_m` already carries the geometry.
- **Candidates:** a ball-unknown run of at least `dead_min_gap_frames` where the players have stopped (gap median of the smoothed walking share and mean speed), the ball seen more than `dead_out_margin_m` outside the pitch, a placed ball (seen, `ball_still_frac` ≥ `dead_placed_still_frac_min`, players stopped), and `known_dead`. A long gap where the players keep running stays alive. A NaN team walking share falls back to `all_walking_frac`; a gap with no player data at all is unknown and is not called live.
- **Boundaries:** a gap ends when the ball is seen again (+ `dead_reappear_offset_frames`, 0). It starts at the line crossing if the ball was last seen out; else at the onset of the stopped run minus `dead_decel_lag_frames`, where the run may reach back at most `dead_onset_max_lookback_frames` before the vanish frame (if it starts earlier, the start is clipped to that bound, with no lag); else at the vanish frame (+ `dead_vanish_offset_frames`, 0). D candidates separated by an alive run shorter than `dead_min_alive_frames` are merged.
- **Restart team:** set-piece tag team (N/A falls through), else the goal tag (the conceding team), else the nearest player at the restart frame, else unknown. Goal-line geometry (goal kick vs corner) is deferred to v2. The tag meanings (the set-piece Team takes the restart; the goals Team scored) are assumptions.
- **Tuning:** `dead_ball.TUNABLE` lists the keys the IDSSE grid may change. The offsets, `dead_out_margin_m`, `dead_onset_max_lookback_frames` and `dead_placed_still_frac_min` are fixed. Harness predictor: `harness.py --predictor dead_ball_v1`.

## Tests (no real data needed)

```bash
python features/tests/make_synthetic_match.py /tmp/synth
MTAH_DATA_DIR=/tmp/synth python features/tests/test_synthetic.py
python features/tests/test_fetch_checks.py      # IDSSE fetch checks
python features/tests/test_idsse_features.py    # IDSSE feature adapter
python features/tests/test_degrade_ball.py      # IDSSE ball-loss simulation
python features/tests/test_harness.py           # IDSSE tuning harness
python features/tests/test_dead_ball.py         # dead-ball detector
python features/tests/test_tune_dead_ball.py    # dead-ball detector tuning
```

This builds a full-length fake match with scripted events and checks the extracted features against the known answers. The events cover:

- a straight ball run, and a short and a long gap
- a 180° reversal
- the ball out of play, and a still ball
- a known nearest player for each team
- one player running

Run it after any change to the extractor.
