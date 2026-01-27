from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from sarif_normalization.normalizator import SASTPreprocessor
from sarif_normalization.post_drop import should_post_drop


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate normalized + post-drop JSON from SARIF.")
    p.add_argument("--oss", required=True, help="OSS project name (used for artifact naming).")
    p.add_argument("--sast", required=True, help="SAST runner name (e.g., semgrep).")
    p.add_argument("--lang", required=True, help="Language label (e.g., python, javascript).")
    p.add_argument("--index", required=True, help="Run index (used for artifact naming).")
    p.add_argument(
        "--sarif",
        default="",
        help="Path to SARIF file. Defaults to <run_name>.sarif in the CWD.",
    )
    p.add_argument(
        "--source-root",
        required=True,
        help="Source code root for snippet extraction.",
    )
    p.add_argument(
        "--snippet-lines",
        type=int,
        default=5,
        help="Number of lines to include around trigger line.",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output JSON path. Defaults to <run_name>.sanitized.json in the CWD.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs.",
    )
    return p.parse_args(argv)


def _build_run_name(oss: str, sast: str, lang: str, index: str) -> str:
    return f"{oss}-{sast}-{lang}-{index}"


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _apply_post_drop(normalized: Dict) -> Dict:
    dropped: List[Dict] = []
    findings: List[Dict] = []

    for finding in normalized.get("findings", []):
        kept_code_units = []
        for code_unit in finding.get("code_units", []):
            reason = should_post_drop(code_unit)
            if reason:
                dropped.append(
                    {
                        "rule_id": finding.get("rule_id", "unknown"),
                        "code_unit_id": code_unit.get("id", ""),
                        "file": code_unit.get("file", ""),
                        "reason": reason,
                    }
                )
                continue
            kept_code_units.append(code_unit)

        if kept_code_units:
            finding["code_units"] = kept_code_units
            findings.append(finding)

    return {"findings": findings, "dropped": dropped}


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    run_name = _build_run_name(args.oss, args.sast, args.lang, args.index)

    run_dir = Path.cwd()
    sarif_path = Path(args.sarif) if args.sarif else (run_dir / f"{run_name}.sarif")
    out_path = Path(args.out) if args.out else (run_dir / f"{run_name}.sanitized.json")

    if not sarif_path.exists():
        print(f"[ERROR] SARIF not found: {sarif_path}")
        return 1

    source_root = Path(args.source_root).expanduser().resolve()
    if not source_root.exists():
        print(f"[ERROR] Source root not found: {source_root}")
        return 1

    pre = SASTPreprocessor(
        sarif_path=str(sarif_path),
        source_root=str(source_root),
        snippet_lines=args.snippet_lines,
    )
    normalized = pre.process()
    post = _apply_post_drop(normalized)
    output = {"findings": post["findings"]}
    _write_json(out_path, output)

    if not args.quiet:
        print(f"[INFO] Final JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
