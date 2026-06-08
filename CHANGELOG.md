# Changelog

All notable changes to VeraRAG are recorded here.

## Unreleased

### Added

- VeraBench report diagnostics: behavior confusion, failure summaries, calibration bins, and conflict TP/FP/FN summaries.
- Offline result analysis CLI: `experiments/analyze_verabench_results.py`.
- VeraBench leaderboard generator: `experiments/build_verabench_leaderboard.py` / `verarag-leaderboard`.
- Calibration curve script compatibility with full VeraBench report JSON.
- Contribution, security, citation, issue, and PR governance files.
- Public `verarag` API package, typed package markers, API/architecture docs, and quickstart example.

### Changed

- Conflict scoring now ignores non-conflict graph edges such as `support` and `partial_support`.
- `pyproject.toml` development dependencies now match the Ruff-based Makefile workflow.
- CI now checks Python 3.10 through 3.13, the public `verarag` package, examples, package build, and full tests.

### Verified

- Full local test suite: `197 passed, 3 skipped` on Python 3.13.9.
