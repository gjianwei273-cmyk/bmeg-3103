# Attention smoothing and threshold search

User approved the proposed joint search, whole-session validation, causal state,
and an accuracy/response balance, with parameter comparison charts.

## Approach

Use the existing Python collector's CSV/JSON pairs without changing acquisition.
Compare raw attention, trailing moving averages (2–10 samples), and EMA
(alpha 0.1–0.9), each with integer thresholds 1–99. Grid search is transparent
and sufficiently small; complex optimizers and EEG feature models are unnecessary.
Python standard library plus a self-contained HTML/SVG/JavaScript report avoids
new dependencies. This implementation choice and scoring weights are defaults.

## Evaluation contract

Only completed serial sessions with eligible valid rows and both labels qualify.
Reject malformed metadata, inconsistent labels, nonfinite values, reordered times,
and duplicate session IDs. Report exclusions. Keep input order and never reset a
filter at a label boundary. Initialize from the first observation of each session.
Missing/preparation data cannot be reconstructed; sample windows are not seconds.

Outer leave-one-session-out evaluation chooses parameters using only the other
sessions. Evaluate the all-method selector and each method family independently,
plus a fixed raw/50 baseline. Macro-average session metrics to avoid long-session
dominance. Report recalls, confusion counts, within-phase flips/minute, and response.

Response is instruction-relative time to confirmation by three consecutive correct
observations within a configurable 15-second horizon. Report the confirmation time,
not the first observation's time. If the last pre-transition prediction was already
the target, do not claim a detected switch: flag it and apply the full horizon as
the delay penalty. Unconfirmed transitions also receive the full horizon. Raw delays
are null for those cases. Report these counts separately. Only observed transitions
are measured; present data has relax→focus only, and preparation samples are absent.

Score = balanced accuracy − 0.005 × mean capped delay seconds
− 0.002 × within-phase flips/minute. Weights, horizon, confirmation count and grids
are CLI-configurable; show them in every report. Prefer higher score, then accuracy,
then less delay, fewer flips, and deterministic parameter order. No minimum score
guarantees real-world generalization. A fresh fixed-parameter session is required.

## Deliverables

Search/metric module, report template, regression tests, README instructions.
Generate timestamped ignored output directories containing report.html,
candidates.csv (all-data exploratory scores), folds.csv (held-out results), and
recommendation.json (deployment config, protocol, provenance, exclusions, validation).
HTML contains held-out comparisons, exploratory tradeoffs/heatmaps, and per-session
causal timelines with raw/smoothed values, threshold and predictions. No CDN/network.
Inputs and unrelated pre-existing working-tree changes are preserved.
