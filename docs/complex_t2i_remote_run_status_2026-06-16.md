# Complex T2I Remote Run Status

Status: RUNNING

Timestamp: 2026-06-16 14:01 Asia/Shanghai

## Run

- Remote host: `lb-gzs` / `wentao@100.69.207.21`
- Remote repo: `/home/wentao/project/Emu3.5`
- Run id: `semantic_uncertainty_complex_t2i_emu35_20260616`
- Run dir: `outputs/semantic_entropy_umm/semantic_uncertainty_complex_t2i_emu35_20260616`
- Concept manifest: `configs/semantic_entropy_t2i_complex_scene_concepts.jsonl`
- Scope: 30 complex T2I concepts x 10 samples = 300 images.
- Atom-level GT template: 30 concepts x 10 samples x 5 checks = 1500 rows.

## GPU Use

The run uses only free/low-occupancy GPU groups:

- worker 0: `CUDA_VISIBLE_DEVICES=0,3`
- worker 1: `CUDA_VISIBLE_DEVICES=5,7`

High-occupancy GPUs `1,2,4,6` were not assigned.

## Current Progress

- T2I generation started at `2026-06-16T11:56:20+08:00`.
- First twenty images completed by about 14:01.
- Completed PNGs: `20/300`.
- Completed concepts/samples:
  - `complex01_chain_three_objects/000.png`, elapsed `776.8s`.
  - `complex01_chain_three_objects/001.png`, elapsed `792.3s`.
  - `complex01_chain_three_objects/002.png`, elapsed `791.0s`.
  - `complex01_chain_three_objects/003.png`.
  - `complex01_chain_three_objects/004.png`.
  - `complex01_chain_three_objects/005.png`.
  - `complex01_chain_three_objects/006.png`.
  - `complex01_chain_three_objects/007.png`.
  - `complex01_chain_three_objects/008.png`.
  - `complex01_chain_three_objects/009.png`.
  - `complex02_vertical_stack_four/000.png`, elapsed `767.9s`.
  - `complex02_vertical_stack_four/001.png`, elapsed `789.2s`.
  - `complex02_vertical_stack_four/002.png`, elapsed `797.3s`.
  - `complex02_vertical_stack_four/003.png`.
  - `complex02_vertical_stack_four/004.png`.
  - `complex02_vertical_stack_four/005.png`.
  - `complex02_vertical_stack_four/006.png`.
  - `complex02_vertical_stack_four/007.png`.
  - `complex02_vertical_stack_four/008.png`.
  - `complex02_vertical_stack_four/009.png`.
- Both workers are still alive and continuing generation.

## First Atom-Level Check

Codex first-pass atom labels were filled for the twenty available images:

- `complex01_chain_three_objects`, sample 0: 5/5 checks pass.
- `complex01_chain_three_objects`, sample 1: 5/5 checks pass.
- `complex01_chain_three_objects`, sample 2: 5/5 checks pass.
- `complex01_chain_three_objects`, sample 3: 5/5 checks pass.
- `complex01_chain_three_objects`, sample 4: 5/5 checks pass.
- `complex01_chain_three_objects`, sample 5: 5/5 checks pass.
- `complex01_chain_three_objects`, sample 6: 5/5 checks pass.
- `complex01_chain_three_objects`, sample 7: 5/5 checks pass.
- `complex01_chain_three_objects`, sample 8: 5/5 checks pass.
- `complex01_chain_three_objects`, sample 9: 5/5 checks pass.
- `complex02_vertical_stack_four`, sample 0: 5/5 checks pass.
- `complex02_vertical_stack_four`, sample 1: 5/5 checks pass.
- `complex02_vertical_stack_four`, sample 2: 5/5 checks pass.
- `complex02_vertical_stack_four`, sample 3: 3/5 checks pass. The image contains an extra red cube below the green cube, so `green cube at bottom` and `exactly four objects` fail.
- `complex02_vertical_stack_four`, sample 4: 5/5 checks pass.
- `complex02_vertical_stack_four`, sample 5: 5/5 checks pass.
- `complex02_vertical_stack_four`, sample 6: 5/5 checks pass.
- `complex02_vertical_stack_four`, sample 7: 5/5 checks pass.
- `complex02_vertical_stack_four`, sample 8: 5/5 checks pass.
- `complex02_vertical_stack_four`, sample 9: 5/5 checks pass.

Partial GT files:

- `outputs/semantic_entropy_umm/semantic_uncertainty_complex_t2i_emu35_20260616/atom_check_annotation/complex_t2i_atom_check_gt_partial_codex.csv`
- `outputs/semantic_entropy_umm/semantic_uncertainty_complex_t2i_emu35_20260616/atom_check_annotation/complex_t2i_atom_check_gt_partial_codex_summary.csv`

## Atom-GT Analysis Readiness

The atom-level GT to uncertainty analysis script is ready and smoke-tested:

- Script: `scripts/analyze_complex_t2i_atom_gt_uncertainty.py`
- Smoke input: partial Codex atom GT for available images.
- Smoke output dir: `outputs/semantic_entropy_umm/semantic_uncertainty_complex_t2i_emu35_20260616/atom_check_annotation/atom_uncertainty_partial`
- Smoke status: `WAITING_FOR_UNCERTAINTY_DATAPOINTS`, as expected because strict/QA postprocessing has not yet produced `t2i_uncertainty_datapoints.csv`.
- Partial coverage: 20 images, 100 atom checks, 1 failed image, 2 failed atom checks.
- Current partial concept labels:
  - `complex01_chain_three_objects`: 10 annotated images, 0 failed images, 0/50 failed checks.
  - `complex02_vertical_stack_four`: 10 annotated images, 1 failed image, 2/50 failed checks.

## Remaining Work

- Let T2I generation continue to 300 images.
- Regenerate atom-level contact sheets as images accumulate.
- Fill atom-level visual GT for all generated images.
- Run strict/QA postprocessing after generation completes.
- Recompute semantic entropy vs true visual hallucination with atom-level labels using `scripts/analyze_complex_t2i_atom_gt_uncertainty.py`.
