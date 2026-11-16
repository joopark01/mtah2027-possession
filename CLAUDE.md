# MTAH 2027: possession per frame (Brian's entry)

Fork of the organisers' repo (`upstream` = metrica-sports/more-than-a-hack-2027, `origin` = joopark01/mtah2027-possession, private until submission). Task: label every tactical-video frame of BAR-ATH as H/A/D, scored by spellF1. Deadline Nov 17, 2026.

## Hard rules (organisers' clarification, 2026-11-03; overrides the repo docs)
- **Never read, print or summarise rows of anything under `data/`.** Claude Code is an AI service, and uploading the provided data to AI services is forbidden. Run scripts that read the data and look only at their aggregate summaries. Never `cat`/`head` data files or open the parquet/npz/csv outputs. `.claude/settings.json` blocks direct reads, but it can't block everything.
- **No training, fitting or tuning on the ground truth or the video.** Every parameter is published, follows from the Laws of the Game or physics, is chosen a priori, or is tuned on **external** data only (IDSSE). `features/config.json` `_provenance` records which, and `check_provenance()` enforces it.
- No hand labels on BAR-ATH, and no checking our output against the video.
- **IDSSE split (fixed 2026-11-09, before any results):** tune on J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ. Hold out J03WPY, J03WR9: evaluate on them only, never tune or select on them.
- **Exception:** external data lives in `external_data/` (gitignored, outside `data/` on purpose) and may be read normally. Never put external data under `data/`.

## Where things are
- `features/extract_features.py` + `config.json`: per-frame feature table → `data/derived/features_tactical.parquet`. `--report` prints a label-free sanity summary.
- `features/diagnose_ball_gaps.py`: what happens while the ball is missing (aggregates only).
- `features/external/fetch_idsse.py`: downloads the 7 IDSSE matches (CC BY 4.0, Bassek et al. 2025) to `external_data/idsse/` as compact npz files with per-frame ball alive/dead and owner labels. The raw XML is deleted after conversion.
- `features/external/idsse_features.py`: same extractor code on each IDSSE npz → `external_data/idsse_features/<id>.parquet`, plus `label` (DFL BallStatus/BallPossession → H/A/D) and `frame_id`. `frame` is a row index there, and the periods are adjacent, so split on `half`.
- `features/tests/`: synthetic-match tests. Run them after any change to the extractor:
  `python features/tests/make_synthetic_match.py /tmp/synth && MTAH_DATA_DIR=/tmp/synth python features/tests/test_synthetic.py && python features/tests/test_fetch_checks.py && python features/tests/test_idsse_features.py && python features/tests/test_degrade_ball.py && python features/tests/test_harness.py && python features/tests/test_dead_ball.py && python features/tests/test_tune_dead_ball.py`

## State (2026-11-15)
- **Feature table built on real data; parsing verified.** Ball seen 59.3%, tracked 20.49, matching DATA.md.
- **Missing ball = mostly stoppages.** The ball is missing in long blocks (132 runs over 5 s, holding 99.7% of missing frames). These are mostly stoppages: players walk, set pieces follow, and the ball reappears moving at the restart.
- **IDSSE feature tables built (7 matches).** The kickoff is the first alive row of each period; the signs match kloppy. Ball kinematics are masked at every block edge.
- **Ball-loss simulation on IDSSE** (`features/external/degrade_ball.py`): a priori values (user's decision, 2026-11-15; not fitted to BAR-ATH). The 21 degraded tables in `external_data/idsse_degraded/` were rebuilt with them (onset/reacquisition lags drawn per dead span). The earlier BAR-ATH-fitted values are kept, unused, in `features/external/ball_loss_fitted_bar_ath.json` until the organisers confirm such fitting is allowed. Use them with `--ball-loss-params <file>`.
- **Tuning harness** (`features/external/harness.py`): scores predictors on IDSSE with `pm.evaluate` and an X at the period seam. Held-out runs need `--holdout` and are logged to `external_data/harness/holdout_log.json`. `baseline_nearest` gets spell_f1_mean about 0.10 on tune data, degraded or clean (difference within noise). It predicts 4–4.6x too many spells. Degraded-table dead-ball edges are now randomised a priori per span (onset lag 0–4 s, reacquisition 0–1 s); a detector's ball vanish/reappear offsets stay `a_priori` and are never tuned. Residual risk: the spreads are themselves assumptions (`docs/findings.md`).
- **Dead-ball detector v1** (`features/dead_ball.py`): rule-based per the design doc. Player-based timing/thresholds tuned on degraded IDSSE tuning tables (`features/external/tune_dead_ball.py`, log in `external_data/dead_ball_tune/`); ball vanish/reappear offsets a priori. In-sample: alive_dead_f1 0.95, spell_f1 0.12 (surface flat; team call is the limit). Transfer risk: IDSSE player tracking is clean (`docs/findings.md`).
- **Detector held-out + BAR-ATH plausibility done** (`docs/findings.md`): held-out alive_dead_f1 0.950; BAR-ATH D share 42.0% excl. pre-kickoff, 144 D segments (too many; 1 s minimum D length proposed, undecided). Held-out pair has had two looks.
- **Next:** possession decoder with the contested state.

## Subagents (`.claude/agents/`)
- **runner:** Bash only. Runs scripts on the provided match and reports aggregate output. Use it for every command that touches `data/`.
- **external-data:** IDSSE fetch, adapters, degradation, tuning. Owns `external_data/` and `features/external/`.
- **implementer:** one pipeline module at a time, with provenance and synthetic tests.
- **reviewer:** read-only rule and science review. Run it after every change, before committing.

Workflow: implementer or external-data writes → reviewer checks → runner runs on real data → main session commits.

## Working with the parallel claude.ai session
- **This repo on the Mac is the single source of truth.** The claude.ai chat sends finished files into it; don't keep two diverging copies.
- **Edit different files at the same time.** Pull new files in with git status / diff before editing.
- **Commit small and often.** The cloud session can't push to `origin`, so commits always happen here.
