# Attention Search Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution. User requested implementation in the current chat.

**Goal:** Reproducible causal attention parameter search and offline comparison report.
**Architecture:** Standard-library Python computes cached candidate/session metrics,
outer session validation, and JSON/CSV outputs; bundled HTML renders SVG charts.
**Tech Stack:** Python 3.9+, unittest, browser JavaScript/SVG.
**Spec:** docs/superpowers/specs/2026-11-07-attention-search-design.md

## Global Constraints

- Preserve acquisition code, input data and existing unrelated working changes.
- Never use a held-out session to select its parameters or labels to reset filters.
- Commit completed changes and add relevant tests; run tests before delivery.
- Label full-data scores exploratory and outer validation separately.

## Review Focus

- Missing/bad CSV metadata and simulation must never silently enter validation.
- A pre-existing target prediction must not masquerade as instant detection.
- Large gaps and irregular arrival times must remain visible in timelines.
- HTML must escape data and work offline, including at small viewport widths.
- Default reruns must preserve earlier reports; small inputs must fail clearly.

## Task 1: Search and report

Files: eeg_collection/tune_attention.py, attention_report.html,
test_tune_attention.py, README.md, .gitignore.

- [x] Write tests for causal raw/MA/EMA prefixes, continuous phase state, boundary
  thresholds, known confusion counts, persistent confirmation and missed switches.
- [x] Write ingestion fixtures covering simulation, incomplete sessions, unavailable
  attention, malformed labels/times and duplicate session IDs.
- [x] Write held-out isolation test: changing held-out labels leaves its choice fixed.
- [x] Write CLI integration test checking real output schemas, baseline, exclusions,
  safe embedded data and rerun preservation; watch tests fail before implementation.
- [x] Implement load_sessions, smooth, evaluate, select, analyze, write_outputs, main;
  use cached metrics to evaluate each candidate/session only once.
- [x] Implement offline report and document commands, scoring and limitations.
- [x] Run full Python suite, real-data CLI, JavaScript syntax/DOM behavior checks,
  and visual browser inspection if available; review the diff and fix findings.
- [x] Commit only task files after verification, record actual results here.

## Execution notes

Work stays in the user-specified checkout on codex/attention-search; existing deleted
tests and untracked hardware resources are unrelated and will not be staged.

Independent review found inconsistent elapsed/phase clock origins could bias delays;
added a rejecting regression (observed failing) and a 2 ms rounding tolerance check.
Added explicit baseline comparison advice: default search does not improve the held-out
score, so the report advises retaining raw/50 rather than claiming a smoothing win.
Test scratch files use the collector's ignored data/test_tmp directory because the
sandbox's default Windows temporary directory denied fixture file writes.
Browser preview could not initialize (Statsig timeout); Node tests run actual report
JavaScript with a minimal DOM to verify plots, controls, finite coordinates and filters.

Final verification: 22/22 Python tests (including Node report integration), 5/5 Node
checks against the generated real-data report, Python bytecode compilation, existing
Arduino executable regression harness, and git diff whitespace check passed.
Real-data CLI: 6 serial sessions, 719 eligible samples, 1 simulation excluded, 1,881
candidates. Held-out selected BA 90.42%, delay penalty 3.794 s, flips/min 2.6285;
fixed raw/50 BA 91.81%, delay penalty 3.2863 s, flips/min 3.2211.
Full-data calibration candidate EMA alpha .7 threshold 47 is explicitly exploratory.
No input data or acquisition code was changed; outputs remain ignored under reports/.
