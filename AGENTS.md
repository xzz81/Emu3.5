# AGENTS.md


## A6000 Emu3.5 Runtime

- On the 4x A6000 server, prefer two GPUs per Emu3.5 model process when running throughput-oriented inference or evaluation. One model can run sharded over two GPUs; with four GPUs available, run two model processes in parallel, e.g. `CUDA_VISIBLE_DEVICES=0,1` and `CUDA_VISIBLE_DEVICES=2,3`, instead of one serial four-GPU process when comparing multiple prompts or configs.


## Entropy Comparison Scope

- Entropy comparisons are only meaningful within the same generated response. Entropy under the ordinary definition must not be compared across different responses or across modalities.
