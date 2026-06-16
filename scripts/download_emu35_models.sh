#!/usr/bin/env bash
set -euo pipefail

mkdir -p model

find_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo "[ERROR] python3 or python is required when huggingface-cli is unavailable" >&2
  return 1
}

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

  local python_bin
  python_bin="$(find_python)"
  "$python_bin" - "$repo_id" "$target_dir" <<'PY'
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
