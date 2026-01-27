
#!/usr/bin/env python3
"""
사용 예:
  python semgrep_runner.py --config p/security-audit --config p/secrets --target .
  python semgrep_runner.py -c p/ci -c p/secrets --jobs 8 --timeout 30 --timeout-threshold 5

주의:
- 이 스크립트는 'semgrep' CLI가 PATH에 있어야 실행됩니다.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


# =========================
# ✅ 기본값(나중에 수정할 부분)
# =========================

# (4) out은 argument로 받지 않고 디폴트로 고정
# TODO: 너 프로젝트 구조에 맞게 저장 위치를 바꿔도 됨
DEFAULT_OUT_DIR = Path("./out")  # 예: ./artifacts/semgrep 등으로 변경 가능
DEFAULT_SARIF_PATH = DEFAULT_OUT_DIR / "semgrep.sarif"

# (3) target default 지정
# TODO: 기본 스캔 경로를 바꾸고 싶으면 여기만 수정
DEFAULT_TARGET = "."  # 보통 프로젝트 루트에서 실행하므로 "."이 무난
DEFAULT_PREDROP_IGNORE = Path(__file__).resolve().parent / "pre-drop.ignore"

# Auto-exclude artifacts
DEFAULT_DRYRUN_JSON = DEFAULT_OUT_DIR / "dryrun_auto.json"
DEFAULT_EXCLUDE_TXT = DEFAULT_OUT_DIR / "exclude_rules.txt"

# Auto-exclude policy/script fixed paths
RUNNER_DIR = Path(__file__).resolve().parent
EXCLUDE_DIR = RUNNER_DIR / "exclude_rules"
DEFAULT_POLICY_PATH = EXCLUDE_DIR / "exclude_policy.yml"
DEFAULT_MAKER_PATH = EXCLUDE_DIR / "make_exclude_rules.py"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run semgrep scan and save SARIF report (fixed output format)."
    )

    # (2) config는 여러 개 입력받아야 함
    p.add_argument(
        "-c",
        "--config",
        action="append",
        required=True,
        help=(
            "Semgrep config (ruleset). Can be specified multiple times. "
            "Examples: -c p/security-audit -c p/secrets, or -c ./rules"
        ),
    )

    # (3) target은 입력받되 default도 지정
    p.add_argument(
        "-t",
        "--target",
        default=DEFAULT_TARGET,
        help=(
            "Scan target path. Default is set in code (DEFAULT_TARGET). "
            "Change DEFAULT_TARGET in the script if you want a different default."
        ),
    )

    # (6) 병렬도 + 리소스 제한만 argument로 받기
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,  # TODO: 기본 병렬도 수정 가능
        help="Number of parallel jobs (semgrep -j). Default: 8 (change in code if needed).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=0,  # 0 = no timeout (semgrep semantics)
        help="Per-rule per-file timeout in seconds. 0 disables timeout. Default: 0.",
    )
    p.add_argument(
        "--timeout-threshold",
        type=int,
        default=0,  # 0 = unlimited threshold (semgrep semantics)
        help="Number of timeouts allowed per file before skipping it. 0 disables threshold. Default: 0.",
    )
    p.add_argument(
        "--max-target-bytes",
        type=int,
        default=0,  # 0 = unlimited (semgrep semantics)
        help="Maximum target file size in bytes. 0 disables the limit. Default: 0.",
    )
    p.add_argument(
        "--pro",
        action="store_true",
        help="Use the Semgrep Pro engine (requires Semgrep Pro auth).",
    )

    # 편의: 실행 커맨드 출력 여부
    p.add_argument(
        "--print-cmd",
        action="store_true",
        help="Print the semgrep command before running it.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress semgrep stdout/stderr (still writes SARIF file).",
    )

    return p.parse_args(argv)


def ensure_semgrep_exists() -> None:
    """Fail fast if semgrep is not available."""
    try:
        subprocess.run(["semgrep", "--version"], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print("[ERROR] 'semgrep' command not found. Install semgrep and ensure it's on PATH.", file=sys.stderr)
        sys.exit(127)
    except subprocess.CalledProcessError as e:
        print("[ERROR] semgrep exists but failed to run '--version'.", file=sys.stderr)
        print(e.stderr or str(e), file=sys.stderr)
        sys.exit(2)


def load_predrop_patterns(ignore_path: Path) -> list[str]:
    if not ignore_path.exists():
        return []

    patterns: list[str] = []
    for raw in ignore_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns
      
# exclude_rules_txt 읽기
def read_exclude_ids(exclude_txt: Path) -> list[str]:
    if not exclude_txt.exists():
        return []
    ids: list[str] = []
    for line in exclude_txt.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.append(s)
    # de-dup preserve order
    seen = set()
    out = []
    for rid in ids:
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out

# dryrun 커맨드 만들기
def build_semgrep_dryrun_command(args: argparse.Namespace, out_json: Path) -> list[str]:
    cmd: list[str] = ["semgrep", "scan"]
    cmd += ["--config", "auto"]
    cmd.append(args.target)
    cmd += ["--dryrun", "--json", "--output", str(out_json)]
    cmd += ["-j", str(args.jobs)]
    cmd += ["--timeout", str(args.timeout)]
    cmd += ["--timeout-threshold", str(args.timeout_threshold)]
    cmd += ["--max-target-bytes", str(args.max_target_bytes)]
    return cmd


def build_semgrep_dryrun_auto_only(args: argparse.Namespace, out_json: Path) -> list[str]:
    return build_semgrep_dryrun_command(args, out_json)

# make_exclude_rules.py
def run_make_exclude_rules(dryrun_json: Path, policy_path: Path, maker_path: Path, out_txt: Path) -> int:
    cmd = [
        "python3",
        str(maker_path),
        "--dryrun-json",
        str(dryrun_json),
        "--policy",
        str(policy_path),
        "--out",
        str(out_txt),
    ]
    proc = subprocess.run(cmd, text=True)
    return proc.returncode



def build_semgrep_command(args: argparse.Namespace, sarif_path: Path, exclude_ids: list[str] | None = None) -> list[str]:
    cmd: list[str] = ["semgrep", "scan"]

    # configs (multiple)
    for c in args.config:
        cmd += ["--config", c]

    # exclude rules(expand)
    if exclude_ids:
        for rid in exclude_ids:
            cmd += ["--exclude-rule", rid]

    # target path
    cmd.append(args.target)

    # fixed output: sarif only
    cmd += ["--sarif", "--sarif-output", str(sarif_path)]

    # pre-drop excludes
    for pattern in load_predrop_patterns(DEFAULT_PREDROP_IGNORE):
        cmd += ["--exclude", pattern]

    # performance / resource controls
    cmd += ["-j", str(args.jobs)]
    cmd += ["--timeout", str(args.timeout)]
    cmd += ["--timeout-threshold", str(args.timeout_threshold)]
    cmd += ["--max-target-bytes", str(args.max_target_bytes)]
    if args.pro:
        cmd.append("--pro")

    return cmd


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ensure_semgrep_exists()

    # (1) 저장 경로는 임의로 잡아둠 (나중에 수정)
    # 결과 폴더 생성
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    sarif_path = DEFAULT_SARIF_PATH

    #Auto-exclude 준비 : dryrun -> exclude_rules.txt
    exclude_ids: list[str] = []

    auto_in_configs = any(c == "auto" for c in args.config)
    auto_exclude_possible = auto_in_configs and DEFAULT_POLICY_PATH.exists() and DEFAULT_MAKER_PATH.exists()

    if auto_exclude_possible:
      dryrun_cmd = build_semgrep_dryrun_auto_only(args, DEFAULT_DRYRUN_JSON)
      if args.print_cmd and not args.quiet:
        print("[INFO] Running(dryrun-auto-only):", shlex.join(dryrun_cmd))

      proc_dry = subprocess.run(dryrun_cmd, text=True)

      if proc_dry.returncode == 0 and DEFAULT_DRYRUN_JSON.exists():
          rc = run_make_exclude_rules(
              dryrun_json=DEFAULT_DRYRUN_JSON,
              policy_path=DEFAULT_POLICY_PATH,
              maker_path=DEFAULT_MAKER_PATH,
              out_txt=DEFAULT_EXCLUDE_TXT,
        )
          if rc == 0:
              exclude_ids = read_exclude_ids(DEFAULT_EXCLUDE_TXT)
              print(f"[INFO] Auto-exclude enabled (auto-only). exclude ids: {len(exclude_ids)}")
          else:
              print("[WARN] make_exclude_rules.py failed. Fallback to normal scan.", file=sys.stderr)
      else:
          print("[WARN] semgrep dryrun(auto) failed. Fallback to normal scan.", file=sys.stderr)

    cmd = build_semgrep_command(args, sarif_path, exclude_ids=exclude_ids)

    if args.print_cmd and not args.quiet:
        print("[INFO] Running:", shlex.join(cmd))

    # 실행
    # 참고: semgrep는 finding이 있어도 기본 exit code가 실패가 아닐 수 있음.
    # 너 파이프라인에서는 LLM verdict로 최종 실패 여부를 결정할 거라서,
    # 여기서는 semgrep exit code를 그대로 반환만 해둠.
    try:
        if args.quiet:
            proc = subprocess.run(
                cmd, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            proc = subprocess.run(cmd, text=True)
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user.", file=sys.stderr)
        return 130

    # 결과 파일 존재 체크
    if not args.quiet:
        if sarif_path.exists():
            print(f"[INFO] SARIF saved to: {sarif_path}")
        else:
            print(f"[WARN] SARIF file not found at: {sarif_path}", file=sys.stderr)

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
