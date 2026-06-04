#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

"""Discover seed-sweep CSVs, merge them, and rebuild the hue benchmark manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys


DEFAULT_EXCLUDES = ("smoke", "generation_hue_filtered_benchmark_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/ume_main_t2i_counterfactual_pairs_hueonly_seed69.py")
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--sweep-root", default="outputs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exclude-substring", action="append", default=list(DEFAULT_EXCLUDES))
    parser.add_argument("--min-generated-tokens", type=int, default=16)
    parser.add_argument("--suggested-seed-count", type=int, default=1)
    parser.add_argument("--suggested-max-total-seconds", type=float, default=2400.0)
    parser.add_argument("--suggested-max-new-tokens", type=int, default=320)
    parser.add_argument("--suggested-per-seed-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--suggested-hard-timeout-seconds", type=float, default=2700.0)
    parser.add_argument("--no-require-image-exists", action="store_true")
    return parser.parse_args()


def should_exclude(path: Path, excludes: list[str]) -> bool:
    lowered = str(path).lower()
    return any(part.lower() in lowered for part in excludes)


def discover_csvs(root: Path, excludes: list[str]) -> list[Path]:
    paths = []
    for path in root.rglob("clean_hue_seed_sweep*.csv"):
        if should_exclude(path, excludes):
            continue
        paths.append(path)
    return sorted(paths)


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["source_csvs"] = str(path)
    return rows


def row_key(row: dict) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("sample_id", "")),
        str(row.get("seed_idx", "")),
        str(row.get("seed", "")),
        str(row.get("generation_mode", "")),
        str(row.get("classifier_free_guidance", "")),
    )


def merge_rows(csv_paths: list[Path]) -> list[dict]:
    merged: dict[tuple[str, str, str, str, str], dict] = {}
    for path in csv_paths:
        for row in read_csv(path):
            key = row_key(row)
            if key in merged:
                sources = [merged[key].get("source_csvs", ""), row.get("source_csvs", "")]
                merged[key]["source_csvs"] = ";".join(source for source in sources if source)
                continue
            merged[key] = row
    return sorted(merged.values(), key=lambda row: (row.get("pair_id", ""), row.get("sample_id", ""), int(float(row.get("seed_idx", 0) or 0))))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_discovery_report(out_dir: Path, args: argparse.Namespace, csv_paths: list[Path], rows: list[dict]) -> None:
    lines = [
        "# Generation Hue Sweep Discovery",
        "",
        f"- Sweep root: `{args.sweep_root}`",
        f"- Excludes: {args.exclude_substring}",
        f"- Discovered CSVs: {len(csv_paths)}",
        f"- Merged unique rows: {len(rows)}",
        f"- Next command: `{out_dir / 'next_sweep_command.sh'}`",
        f"- All commands: `{out_dir / 'all_sweep_commands.sh'}`",
        f"- Run next and refresh: `{out_dir / 'run_next_and_refresh.sh'}`",
        "",
        "## CSV Inputs",
        "",
    ]
    for path in csv_paths:
        lines.append(f"- `{path}`")
    (out_dir / "sweep_discovery_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_manifest(args: argparse.Namespace, combined_csv: Path) -> None:
    command = [
        sys.executable,
        "scripts/build_generation_hue_benchmark_manifest.py",
        "--cfg",
        args.cfg,
        "--max-pairs",
        str(args.max_pairs),
        "--csv",
        str(combined_csv),
        "--out-dir",
        args.out_dir,
        "--min-generated-tokens",
        str(args.min_generated_tokens),
        "--suggested-seed-count",
        str(args.suggested_seed_count),
        "--suggested-max-total-seconds",
        str(args.suggested_max_total_seconds),
        "--suggested-max-new-tokens",
        str(args.suggested_max_new_tokens),
        "--suggested-hard-timeout-seconds",
        str(args.suggested_hard_timeout_seconds),
        "--suggested-per-seed-timeout-seconds",
        str(args.suggested_per_seed_timeout_seconds),
    ]
    if args.no_require_image_exists:
        command.append("--no-require-image-exists")
    subprocess.run(command, check=True)


def refresh_command(args: argparse.Namespace) -> str:
    parts = [
        "python",
        "scripts/build_generation_hue_benchmark_from_sweeps.py",
        "--cfg",
        args.cfg,
        "--max-pairs",
        str(args.max_pairs),
        "--sweep-root",
        args.sweep_root,
        "--out-dir",
        args.out_dir,
        "--min-generated-tokens",
        str(args.min_generated_tokens),
        "--suggested-seed-count",
        str(args.suggested_seed_count),
        "--suggested-max-total-seconds",
        str(args.suggested_max_total_seconds),
        "--suggested-max-new-tokens",
        str(args.suggested_max_new_tokens),
        "--suggested-hard-timeout-seconds",
        str(args.suggested_hard_timeout_seconds),
        "--suggested-per-seed-timeout-seconds",
        str(args.suggested_per_seed_timeout_seconds),
    ]
    if tuple(args.exclude_substring) != DEFAULT_EXCLUDES:
        for exclude in args.exclude_substring:
            parts.extend(["--exclude-substring", exclude])
    if args.no_require_image_exists:
        parts.append("--no-require-image-exists")
    return " ".join(parts)


def write_command_files(out_dir: Path, args: argparse.Namespace) -> None:
    queue_path = out_dir / "missing_sweep_queue.csv"
    rows = read_csv(queue_path)
    commands = [row.get("suggested_command", "") for row in rows if row.get("suggested_command")]
    next_path = out_dir / "next_sweep_command.sh"
    all_path = out_dir / "all_sweep_commands.sh"
    run_next_refresh_path = out_dir / "run_next_and_refresh.sh"
    header = "#!/usr/bin/env bash\nset -euo pipefail\n\n"
    refresh = refresh_command(args)
    if commands:
        next_path.write_text(header + commands[0] + "\n", encoding="utf-8")
        all_path.write_text(header + "\n".join(commands) + "\n", encoding="utf-8")
        run_next_refresh_path.write_text(
            header
            + "mkdir -p "
            + str(out_dir / "logs")
            + "\n"
            + "LOG_PATH=\""
            + str(out_dir / "logs")
            + "/run_next_$(date +%Y%m%d_%H%M%S).log\"\n"
            + "echo \"[INFO] logging to ${LOG_PATH}\"\n"
            + "{\n"
            + "  echo \"[INFO] sweep started at $(date -Is)\"\n"
            + "  set +e\n"
            + "  "
            + commands[0]
            + "\n"
            + "  SWEEP_STATUS=$?\n"
            + "  set -e\n"
            + "  echo \"[INFO] sweep exit status ${SWEEP_STATUS} at $(date -Is)\"\n"
            + "  echo \"[INFO] refresh started at $(date -Is)\"\n"
            + "  "
            + refresh
            + "\n"
            + "  echo \"[INFO] done at $(date -Is)\"\n"
            + "  exit ${SWEEP_STATUS}\n"
            + "} 2>&1 | tee \"${LOG_PATH}\"\n",
            encoding="utf-8",
        )
    else:
        next_path.write_text(header + "echo 'No missing sweep commands.'\n", encoding="utf-8")
        all_path.write_text(header + "echo 'No missing sweep commands.'\n", encoding="utf-8")
        run_next_refresh_path.write_text(header + refresh + "\n", encoding="utf-8")
    next_path.chmod(0o755)
    all_path.chmod(0o755)
    run_next_refresh_path.chmod(0o755)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = discover_csvs(Path(args.sweep_root), args.exclude_substring)
    rows = merge_rows(csv_paths)
    combined_csv = out_dir / "clean_hue_seed_sweeps_merged.csv"
    write_csv(combined_csv, rows)
    write_discovery_report(out_dir, args, csv_paths, rows)
    run_manifest(args, combined_csv)
    write_command_files(out_dir, args)
    print(f"[INFO] discovered {len(csv_paths)} CSVs; merged {len(rows)} rows; manifest saved to {out_dir}")


if __name__ == "__main__":
    main()
