# Broadcast-to-tactical alignment: design v1 (2026-11-16, diagnostic only)

**Goal:** map each broadcast ATD frame to a tactical video frame (or none) from tracking alone, so that we can measure whether the broadcast sees the ball where the tactical ATD loses it. v1 feeds nothing: the decoder and the submission don't read its output.

**Code:** `features/broadcast_align.py` (core `align()` is a pure function on arrays), test `features/tests/test_broadcast_align.py`.
**Output:** `data/derived/broadcast_alignment.parquet` (broadcast_frame, tactical_frame or -1, segment_id, is_replay, match_dist_m). `--report` prints label-free aggregates only.

## Inputs
- Both ATDs, positions only (`ef.parse_raw_atd` / `ef.load_arrays`), scaled to metres by `games.GAMES["BAR-ATH"]["pitch"]`. Both use normalised 0–1 pitch coordinates (DATA.md). Both metadata files must report `GlobalConfig/FrameRate` = config `fps` (25). This is asserted.
- **No slot identities.** The two ATDs come from separate tracker runs (46 vs 51 slots), and slots aren't persons (DATA.md). Matching is therefore assignment-free. Team membership comes from each file's metadata (first Team = home). The report checks that the team names agree.
- For the report only: `features_tactical.parquet`, and `dead_ball.detect` with `harness.frozen_config()` and `dead_ball_report.bar_ath_ctx`.

## Method
1. **Match distance** d(b, t): the median, over the broadcast players at frame b, of the distance to the nearest same-team tactical player at frame t. It is one-directional because the broadcast shows only part of the pitch. The median tolerates a few spurious broadcast tracks.
2. **Coarse search (1 Hz):** every 25th broadcast frame is compared with every 25th tactical frame that has enough players. The top-k local minima with d ≤ coarse threshold become candidate offsets (t − b).
3. **Refine:** for each broadcast frame with ≥ min_players, try the candidate offsets of its two bracketing coarse samples, ±12 frames each. Keep the best t and its d.
4. **Segments (constant offset):** a segment starts at a frame whose best d ≤ max_dist and keeps that offset while d(b, b + o) ≤ max_dist, players stay tracked and t stays in one tactical half block. A segment never spans a half boundary. The offset is then refitted over the whole segment (±12 frames, minimum of the capped distance sum), and the segment is split wherever the refitted d exceeds max_dist. Segments shorter than 1 s are dropped, so there are no blips. Abutting segments with the same offset are merged. Two properties follow:
   - A cut that keeps the offset gives two segments with one offset.
   - In a stoppage the offset is held from the motion around it. Per-frame matching is ambiguous there, so the offset is never re-chosen from those frames alone.
5. **Live vs replay:** the live map is the maximum-total-length chain of segments whose tactical intervals increase with broadcast time (t_start_j ≥ t_end_i). A segment outside the chain that starts before the tactical time the chain has already reached is a **replay**: it is kept and flagged, and it never enters the live map. Any other segment is dropped. Two invariants are asserted: the live mapping is strictly increasing, and every segment is ≥ min length.

## Parameters (`features/config.json`, all `a_priori`)
None of these can be tuned externally: IDSSE has no broadcast feed, and tuning on BAR-ATH GT or video is forbidden. They stay a priori with no tune log. `frozen_config` still passes.

| key | value | basis |
|---|---|---|
| `broadcast_align_coarse_step_frames` | 25 | 1 Hz. Any true offset is within 12 frames of a sampled tactical frame. |
| `broadcast_align_coarse_top_k` | 3 | Room for the live match plus two look-alikes (a replay source, a similar set-piece shape). |
| `broadcast_align_coarse_max_dist_m` | 3.0 | Fine threshold, plus up to 0.48 s of grid misalignment at 2–3 m/s (~1–1.5 m). |
| `broadcast_align_refine_half_window_frames` | 12 | Half the coarse step, so it covers the grid rounding exactly. |
| `broadcast_align_min_players` | 6 | Close-ups show 1–4 players. A median over so few points matches at many wrong times. |
| `broadcast_align_max_dist_m` | 1.5 | ~0.7 m assumed per-system error (the broadcast camera moves and zooms), ~1 m combined, plus one frame at sprint speed (0.32 m), rounded up. |
| `broadcast_align_min_segment_frames` | 25 | 1 s. TV shots are rarely shorter, and a sub-second segment can't pin an offset. |

## What v1 does not do
- **No slow-motion replays.** A replay at rate ≠ 1 has no constant offset. It breaks into pieces that are mostly too short or too poorly matched, so those frames end up unmapped. That is safe, since replays never feed the live map.
- **No use of the broadcast homography `valid` flag or reprojection error.** The tracked-player count stands in for them. Pixels and video are never used.
- **No identity matching.** Slots aren't persons, and the slot counts differ.
- **No sub-frame offset and no drift model.** Both feeds are 25 fps, which is asserted.
- **Static stoppages.** If a segment *starts* inside a long, nearly static stretch, its seed offset can be wrong. Its refit is then ambiguous too. The error stays within the static stretch, where the positions are near-identical anyway, but it can push out a correct segment in the chain.
- **Orientation and team order are checked, not corrected.** The report prints the coarse match share for four variants of the broadcast (as given, teams swapped, y mirrored, rotated 180°). If "as given" isn't the clear winner, v1's output is meaningless and the convention must be decided before use.

## How it would feed the pipeline (later, not in v1)
- **Ball-gap filling:** on tactical frames where the tactical ball is missing but the live-mapped broadcast frame sees the ball, use the broadcast ball position, after a consistency check (the report's median/p90 broadcast-vs-tactical ball distance where both are seen). Feature columns (nearest team, ball speed, out of play) would then be recomputed from the filled track. Any new thresholds would need IDSSE-free a priori bases, because nothing here can be tuned externally.
- **Dead-ball evidence:** close-ups and replays (unmapped or replay broadcast stretches) are a stoppage cue at the matching live tactical time. Mapping that cue needs the live offset carried across the gap. Segment ends give it.
- Whether either is worth doing depends on the report's key aggregate: how many ball-missing graded tactical frames get a live-mapped broadcast frame with the ball seen, split by the detector's D vs alive call.

## Rule compliance
- **Tracking only:** both ATDs (positions, team membership) and the published frame rate. No video pixels, no homography matrices, no Smart Tagging, no labels or GT.
- **Nothing is fitted to BAR-ATH:** every parameter is a priori (above). The per-frame offsets come from optimising match distance, which is the alignment itself, not tuning of a parameter against labels, and the output is a frame correspondence.
- **Aggregates only:** the report prints counts, shares and quantiles. The parquet stays under `data/derived/` (gitignored) and is never read by the assistant.
