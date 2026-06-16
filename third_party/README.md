# third_party/

Local third-party reference repositories can live here when they are not project code.

- Do not commit cloned repositories or their nested `.git` directories.
- Keep project-facing data entry points under `data/`.
- Keep benchmark or baseline repositories under `external/benchmarks/` when they are part of benchmark execution.
- Document any required third-party clone in the relevant script, report, or registry.
