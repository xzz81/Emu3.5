#!/usr/bin/env bash
set -euo pipefail

mkdir -p model

download_repo() {
  local repo_id="$1"
  local target_dir="$2"

  if [[ -e "$target_dir" ]]; then
    echo "[SKIP] $target_dir already exists"
    return
  fi

  echo "[INFO] downloading $repo_id -> $target_dir"
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo_id" \
      --local-dir "$target_dir" \
      --local-dir-use-symlinks False
    return
  fi

  python - "$repo_id" "$target_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, target_dir = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False,
)
PY
}

download_repo "BAAI/Emu3.5" "model/Emu3.5"
download_repo "BAAI/Emu3.5-VisionTokenizer" "model/Emu3.5-VisionTokenizer"

if [[ "${DOWNLOAD_EMU35_IMAGE:-0}" == "1" ]]; then
  download_repo "BAAI/Emu3.5-Image" "model/Emu3.5-Image"
fi
