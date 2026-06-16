#!/usr/bin/env python3
"""Static claim-boundary audit for semantic entropy docs and reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FORBIDDEN_PATTERNS = [
    "text entropy ~= image entropy",
    "text entropy and image entropy are directly comparable",
    "text entropy and image entropy are directly comparable.",
    "raw token-level entropy across modalities",
    "UMM has a unified internal entropy space",
    "unified internal entropy space",
    "directly compare raw token",
    "直接比较跨模态 token",
    "直接比较原始文本",
    "统一内部熵空间",
]

GUARD_TERMS = [
    "not",
    "cannot",
    "do not",
    "does not",
    "without",
    "forbidden",
    "not allowed",
    "still not allowed",
    "claim boundary",
    "rather than",
    "we do not compare",
    "not prove",
    "不可",
    "不可接受",
    "不能",
    "不能接受",
    "不是",
    "不是证明",
    "禁止",
    "仍然不能",
    "仍然不能接受",
    "不直接",
    "边界",
]

REQUIRED_ALLOWED_PATTERNS = [
    "controlled semantic-state protocol",
    "same finite semantic space",
    "shared semantic state",
    "H(S | concept, route",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--outputs-dir", default="outputs/semantic_entropy_umm")
    parser.add_argument("--out-dir", default="outputs/semantic_entropy_umm/claim_boundary_audit")
    return parser.parse_args()


def markdown_files(docs_dir: Path, outputs_dir: Path) -> list[Path]:
    files = []
    if docs_dir.exists():
        files.extend(sorted(docs_dir.glob("*.md")))
    if outputs_dir.exists():
        files.extend(sorted(outputs_dir.glob("**/*.md")))
    return [path for path in files if "claim_boundary_audit" not in str(path)]


def classify(lines: list[str], idx: int) -> str:
    window_start = max(0, idx - 6)
    window_end = min(len(lines), idx + 6)
    window = " ".join(lines[window_start:window_end]).lower()
    if any(term.lower() in window for term in GUARD_TERMS):
        return "guarded_boundary_mention"
    return "potential_overclaim"


def scan_file(path: Path) -> tuple[list[dict], list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    findings = []
    allowed_hits = []
    for idx, line in enumerate(lines):
        lower = line.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.lower() in lower:
                findings.append(
                    {
                        "file": str(path),
                        "line": idx + 1,
                        "pattern": pattern,
                        "classification": classify(lines, idx),
                        "text": line.strip(),
                    }
                )
        for pattern in REQUIRED_ALLOWED_PATTERNS:
            if pattern.lower() in lower:
                allowed_hits.append(pattern)
    return findings, allowed_hits


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "line", "pattern", "classification", "text"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["(no rows)"]
    fields = ["file", "line", "pattern", "classification", "text"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        safe = {key: str(row.get(key, "")).replace("|", "\\|") for key in fields}
        lines.append("| " + " | ".join(safe[field] for field in fields) + " |")
    return lines


def write_report(out_dir: Path, findings: list[dict], allowed_hits: dict[str, int], files: list[Path]) -> None:
    potential = [row for row in findings if row["classification"] == "potential_overclaim"]
    guarded = [row for row in findings if row["classification"] == "guarded_boundary_mention"]
    status = "PASS" if not potential else "FAIL"
    lines = [
        "# Claim Boundary Audit",
        "",
        f"Status: {status}",
        "",
        "This static audit scans semantic entropy Markdown docs/reports for forbidden claim phrases. It treats explicitly negated or boundary-context mentions as guarded.",
        "",
        "## Input Files",
        "",
        "- `docs/*.md`",
        "- `outputs/semantic_entropy_umm/**/*.md`",
        "",
        "## Commands",
        "",
        "```bash",
        "./.venv-transformers/bin/python scripts/build_semantic_entropy_claim_boundary_audit.py \\",
        "  --docs-dir docs \\",
        "  --outputs-dir outputs/semantic_entropy_umm \\",
        f"  --out-dir {out_dir}",
        "```",
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'claim_boundary_findings.csv'}`",
        f"- `{out_dir / 'claim_boundary_audit_report.md'}`",
        "",
        "## Sample Counts",
        "",
        f"- Markdown files scanned: `{len(files)}`",
        f"- Guarded boundary mentions: `{len(guarded)}`",
        f"- Potential overclaims: `{len(potential)}`",
        f"- Required allowed-claim pattern families observed: `{len(allowed_hits)}`",
        "",
        "## Pass/Fail Checks",
        "",
        f"- Status PASS: `{status == 'PASS'}`",
        f"- Potential overclaims absent: `{not potential}`",
        f"- Markdown files found: `{bool(files)}`",
        "- Static-lint-only caveat retained: `True`",
        "",
        "## Counts",
        "",
        f"- Markdown files scanned: {len(files)}",
        f"- Guarded boundary mentions: {len(guarded)}",
        f"- Potential overclaims: {len(potential)}",
        f"- Required allowed-claim pattern families observed: {len(allowed_hits)}",
        "",
        "## Required Allowed-Claim Hits",
        "",
        *[f"- `{pattern}`: {count}" for pattern, count in sorted(allowed_hits.items())],
        "",
        "## Potential Overclaims",
        "",
        *md_table(potential),
        "",
        "## Guarded Boundary Mentions",
        "",
        *md_table(guarded),
        "",
        "## Claim Allowed After This Step",
        "",
        "The current Markdown evidence package has been checked for obvious forbidden-claim wording.",
        "",
        "## Claim Still Not Allowed",
        "",
        "This lint does not prove the scientific claim; manual extractor validation and full robustness remain required before upgrading beyond extractor-limited evidence.",
    ]
    (out_dir / "claim_boundary_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = markdown_files(Path(args.docs_dir), Path(args.outputs_dir))
    all_findings = []
    allowed_counts: dict[str, int] = {}
    for path in files:
        findings, allowed_hits = scan_file(path)
        all_findings.extend(findings)
        for pattern in allowed_hits:
            allowed_counts[pattern] = allowed_counts.get(pattern, 0) + 1
    write_csv(out_dir / "claim_boundary_findings.csv", all_findings)
    write_report(out_dir, all_findings, allowed_counts, files)
    potential = [row for row in all_findings if row["classification"] == "potential_overclaim"]
    print(
        f"[INFO] scanned {len(files)} markdown files; guarded={len(all_findings) - len(potential)}; potential_overclaims={len(potential)}"
    )
    if potential:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
