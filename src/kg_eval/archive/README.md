# Development outputs

`type-only-development/` holds intermediate runs produced during the 2026-09-07
implementation audit. They use the earlier relation-agnostic development
protocol, not the final semantic evaluator. Some source files were being edited
while those runs finished, so their recorded code hashes are not a reproducible
code snapshot. Do not use these runs as published results or in comparisons.

They were moved here reversibly after automatic approval review rejected their
deletion. The generated development subfolder is ignored by Git. Final validated
results are in `../results/BioHopR/`, under the graph-specific folders ending in
`_semantic_full`. See `../VALIDATION.md` for the exact run links.
