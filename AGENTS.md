# AGENTS.md


## Repository Portability

- Use the repository root as the project root.
- Do not write absolute paths in code, configs, scripts, or notebooks. All paths must be relative to the repository root.
- Datasets must be accessed through `data/`.
- Models, checkpoints, tokenizers, and adapters must be accessed through `model/`.
- If required data or models already exist on the server, use local symlinks under `data/` or `model/` instead of downloading duplicates.
- Do not commit server-specific paths, symlinks, datasets, model weights, checkpoints, logs, or outputs.
- A fresh Git clone should be deployable on another server by recreating the required `data/` and `model/` links, without changing source code.


## A6000 Emu3.5 Runtime

- On the 4x A6000 server, prefer two GPUs per Emu3.5 model process when running throughput-oriented inference or evaluation. One model can run sharded over two GPUs; with four GPUs available, run two model processes in parallel, e.g. `CUDA_VISIBLE_DEVICES=0,1` and `CUDA_VISIBLE_DEVICES=2,3`, instead of one serial four-GPU process when comparing multiple prompts or configs.


## Entropy Comparison Scope

- Entropy comparisons are only meaningful within the same generated response. Entropy under the ordinary definition must not be compared across different responses or across modalities.


## Emu3.5 T2I Baseline

- Keep `./weights/Emu3.5` as the current default T2I baseline unless the task explicitly asks to compare against `Emu3.5-Image`.
- Do not treat the old `16x16` visual grid with `max_new_tokens=360` as a valid T2I result. That setting is only a low-cost smoke test and can produce nearly blank or striped images.
- Formal Emu3.5 T2I runs should use at least `target_height=64`, `target_width=64`, `image_area=1048576`, and `t2i_max_new_tokens=5120`, with the current baseline using `classifier_free_guidance=2.0`, `image_top_k=5120`, and `image_temperature=1.0`.
