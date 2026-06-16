# model/

Project-facing model entry points live here.

- Use this directory for model, checkpoint, tokenizer, and adapter entry points.
- Use symlinks when assets already exist elsewhere on the server.
- Do not commit model weights, checkpoints, tokenizers, adapters, or server-specific symlinks.
- Keep reusable download scripts under `scripts/` and track them with git.
