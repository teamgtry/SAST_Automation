#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ERROR_LOG_NAME = "codeql_runner_error.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a CodeQL database and analyze it to produce SARIF output."
    )
    parser.add_argument(
        "--language",
        required=True,
        help="CodeQL language (e.g., javascript, python, cpp).",
    )
    parser.add_argument(
        "--source-root",
        default=Path(".."),
        type=Path,
        help="Source root for CodeQL extraction (default: parent directory '..').",
    )
    parser.add_argument(
        "--db-out",
        default=Path(".") / "codeql-db",
        type=Path,
        help="Directory to place the CodeQL database (default: ./codeql-db).",
    )
    parser.add_argument(
        "--sarif-out",
        default=None,
        type=Path,
        help="Path to write the SARIF report (default: ./<cwd>.sarif, adding -1, -2... if exists).",
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="CodeQL query suite or pack (default: codeql/<language>-queries).",
    )
    parser.add_argument(
        "--codeql-path",
        default=None,
        help="Explicit path to the CodeQL CLI (default: use CODEQL_PATH env or look in PATH).",
    )

    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        dest="mode",
        action="store_const",
        const="overwrite",
        help="Recreate the database if it already exists (default).",
    )
    overwrite_group.add_argument(
        "--resume",
        dest="mode",
        action="store_const",
        const="resume",
        help="Reuse an existing database without recreating it.",
    )
    parser.set_defaults(mode="overwrite")

    return parser.parse_args()


def find_codeql_binary(explicit_path: str | None) -> str | None:
    candidates: list[str] = []
    env_path = os.environ.get("CODEQL_PATH")

    if explicit_path:
        candidates.append(explicit_path)
    if env_path:
        candidates.append(env_path)

    auto_path = shutil.which("codeql")
    if auto_path:
        candidates.append(auto_path)

    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return str(candidate_path)
        if candidate_path.is_dir():
            binary = candidate_path / "codeql"
            if binary.is_file():
                return str(binary)
    return None


def log_error(error_path: Path, cmd: list[str], result: subprocess.CompletedProcess[str]) -> None:
    error_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines = [
        f"[{timestamp}] Command failed:",
        " ".join(cmd),
        "",
        "stdout:",
        result.stdout or "<empty>",
        "",
        "stderr:",
        result.stderr or "<empty>",
        "",
    ]
    with error_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Error log written to {error_path}", file=sys.stderr)


def run_codeql(cmd: list[str], error_path: Path) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log_error(error_path, cmd, result)
        sys.exit(result.returncode)


def ensure_db_path(db_path: Path, mode: str) -> None:
    if db_path.exists():
        if mode == "resume":
            return
        if db_path.is_dir():
            shutil.rmtree(db_path)
        else:
            db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)


def default_sarif_path() -> Path:
    """Build a SARIF path like ./<cwd>.sarif, adding -1, -2... if already present."""
    base_dir = Path(".")
    base_name = Path.cwd().name
    index = 0
    while True:
        suffix = f"-{index}" if index > 0 else ""
        candidate = base_dir / f"{base_name}{suffix}.sarif"
        if not candidate.exists():
            return candidate
        index += 1


def main() -> None:
    args = parse_args()
    error_log_path = Path.cwd() / ERROR_LOG_NAME

    codeql_binary = find_codeql_binary(args.codeql_path)
    if not codeql_binary:
        print(
            "CodeQL CLI not found. Install from https://aka.ms/codeql-cli and ensure it is on PATH or set CODEQL_PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    suite = args.suite or f"codeql/{args.language}-queries"
    source_root = args.source_root.resolve()
    db_path = args.db_out
    sarif_path = args.sarif_out or default_sarif_path()

    print(f"Using CodeQL CLI: {codeql_binary}")
    print(f"Language: {args.language}")
    print(f"Source root: {source_root}")
    print(f"Database path: {db_path}")
    print(f"SARIF output: {sarif_path}")
    print(f"Query suite: {suite}")

    ensure_db_path(db_path, args.mode)
    sarif_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode != "resume" or not db_path.exists():
        create_cmd = [
            codeql_binary,
            "database",
            "create",
            str(db_path),
            f"--language={args.language}",
            f"--source-root={source_root}",
        ]
        if args.mode == "overwrite":
            create_cmd.append("--overwrite")
        print("Creating CodeQL database...")
        run_codeql(create_cmd, error_log_path)
    else:
        print("Reusing existing database (resume mode).")

    analyze_cmd = [
        codeql_binary,
        "database",
        "analyze",
        str(db_path),
        suite,
        f"--format=sarif-latest",
        f"--output={sarif_path}",
        "--ram=8192",
        "--rerun",
    ]
    print("Running CodeQL analysis...")
    run_codeql(analyze_cmd, error_log_path)

    if not sarif_path.exists():
        dummy_result = subprocess.CompletedProcess(analyze_cmd, returncode=1, stdout="", stderr="SARIF file not created.")
        log_error(error_log_path, analyze_cmd, dummy_result)
        print("SARIF output not found after analysis.", file=sys.stderr)
        sys.exit(1)

    print(f"SARIF created at {sarif_path}")


if __name__ == "__main__":
    main()
