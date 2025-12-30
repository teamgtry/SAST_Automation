#!/usr/bin/env python3
"""
Top-level CLI to choose a SAST runner and execute it with basic arguments.

Flow:
1) Prompt user to select runner (codeql or semgrep).
2) Prompt for language.
3) Prompt for runner-specific options (CodeQL suite or Semgrep configs/target/resources).
4) Prompt for LLM verifier thread count.
5) Create an output directory at ../<oss>-<runner>-<lang>-<index>.
6) Invoke the chosen runner script, passing the language (and SARIF/DB paths where known).
7) Normalize SARIF into JSON and run the LLM verifier (llm_verifier.ts).
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def prompt_runner() -> str:
    options = {"1": "codeql", "2": "semgrep"}
    print("Select runner:")
    print("  1) codeql")
    print("  2) semgrep")
    while True:
        choice = input("> ").strip().lower()
        if choice in options:
            return options[choice]
        if choice in options.values():
            return choice
        print("Please enter 1 or 2 (codeql/semgrep).")


def prompt_language() -> str:
    while True:
        lang = input("Language (e.g., javascript, python): ").strip()
        if lang:
            return lang
        print("Language is required.")


def prompt_codeql_suite(language: str) -> str | None:
    prompt = f"CodeQL suite/pack (default codeql/{language}-queries): "
    raw = input(prompt).strip()
    return raw or None


def prompt_semgrep_configs() -> list[str]:
    """Prompt for one or more Semgrep configs (rulesets)."""
    while True:
        raw = input(
            "Semgrep config(s) (space/comma separated, required): "
        ).strip()
        configs = [part for part in raw.replace(",", " ").split() if part]
        if configs:
            return configs
        print("At least one Semgrep config is required.")


def prompt_semgrep_target(default_target: Path) -> Path:
    prompt = f"Semgrep target path (default {default_target}): "
    raw = input(prompt).strip()
    chosen = Path(raw) if raw else default_target
    return chosen.expanduser().resolve()


def prompt_optional_int(prompt: str, allow_zero: bool = True) -> int | None:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer (or leave blank for default).")
            continue
        if value > 0 or (allow_zero and value >= 0):
            return value
        print("Please enter a non-negative integer." if allow_zero else "Please enter a positive integer.")


def prompt_threads() -> int:
    while True:
        raw = input("Threads for LLM verifier (default 1): ").strip()
        if not raw:
            return 1
        try:
            value = int(raw)
            if value >= 1:
                return value
        except ValueError:
            pass
        print("Please enter a positive integer for threads.")


def is_wsl() -> bool:
    """Detect if running under WSL to avoid using Windows shims like npx.cmd."""
    try:
        return "microsoft" in platform.release().lower()
    except Exception:
        return False


def resolve_npx_binary() -> str:
    """
    Prefer the native npx for the current OS.
    On Windows -> npx.cmd, otherwise -> npx.
    """
    if os.name == "nt" and not is_wsl():
        return "npx.cmd"
    return "npx"


def allocate_run_dir(oss: str, runner: str, language: str) -> tuple[Path, str, int]:
    """
    Find the first available ../<oss>-<runner>-<lang>-<index> directory.
    Returns (run_dir_path, base_name_with_index, index).
    """
    parent = Path.cwd()
    base_prefix = f"{oss}-{runner}-{language}"
    index = 0
    while True:
        base_name = f"{base_prefix}-{index}"
        candidate = parent / base_name
        if not candidate.exists():
            return candidate, base_name, index
        index += 1


def run_codeql(
    language: str, run_dir: Path, run_name: str, suite: str | None = None
) -> int:
    runner_path = Path("sast_runner") / "codeql_runner.py"
    if not runner_path.exists():
        print(f"codeql runner not found at {runner_path}")
        return 1

    sarif_path = run_dir / f"{run_name}.sarif"
    db_path = run_dir / "codeql-db"

    cmd = [
        sys.executable,
        str(runner_path),
        "--language",
        language,
        "--sarif-out",
        str(sarif_path),
        "--db-out",
        str(db_path),
    ]
    if suite:
        cmd.extend(["--suite", suite])
    print(f"Running CodeQL: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_semgrep(
    configs: list[str],
    target: Path,
    run_dir: Path,
    run_name: str,
    jobs: int | None = None,
    timeout: int | None = None,
    timeout_threshold: int | None = None,
    max_target_bytes: int | None = None,
) -> int:
    runner_path = (Path("sast_runner") / "semgrep_runner.py").resolve()
    if not runner_path.exists():
        print(f"semgrep runner not found at {runner_path}")
        return 1

    sarif_path = run_dir / f"{run_name}.sarif"
    runner_default_sarif = run_dir / "out" / "semgrep.sarif"

    cmd = [sys.executable, str(runner_path)]
    for config in configs:
        cmd.extend(["--config", config])
    cmd.extend(["--target", str(target)])
    if jobs is not None:
        cmd.extend(["--jobs", str(jobs)])
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])
    if timeout_threshold is not None:
        cmd.extend(["--timeout-threshold", str(timeout_threshold)])
    if max_target_bytes is not None:
        cmd.extend(["--max-target-bytes", str(max_target_bytes)])

    print(f"Running Semgrep: {' '.join(cmd)} (cwd={run_dir})")
    result = subprocess.run(cmd, cwd=run_dir)
    if result.returncode != 0:
        return result.returncode

    if runner_default_sarif.exists():
        try:
            runner_default_sarif.replace(sarif_path)
        except OSError as exc:  # pragma: no cover - simple filesystem failure path
            print(f"Failed to move SARIF file: {exc}", file=sys.stderr)
            return 1
        print(f"SARIF saved to: {sarif_path}")
    else:
        print(f"SARIF file not found at expected location: {runner_default_sarif}")
    return result.returncode


def run_generate_json(
    oss: str, runner: str, language: str, index: int, run_dir: Path
) -> int:
    project_root = Path(__file__).resolve().parent
    script_path = project_root / "sarif_normalization" / "generate_json.py"
    if not script_path.exists():
        print(f"generate_json not found at {script_path}")
        return 1

    env = os.environ.copy()
    # Ensure project root is on PYTHONPATH so package imports work even when cwd is run_dir
    existing_pp = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pp}" if existing_pp else str(project_root)
    )

    cmd = [
        sys.executable,
        str(script_path),
        "--oss",
        oss,
        "--sast",
        runner,
        "--lang",
        language,
        "--index",
        str(index),
    ]
    print(f"Generating normalized JSON: {' '.join(cmd)} (cwd={run_dir})")
    result = subprocess.run(cmd, cwd=run_dir, env=env)
    return result.returncode


def run_llm_verifier(run_dir: Path, run_name: str, threads: int) -> int:
    project_root = Path(__file__).resolve().parent
    repo_root = project_root.parent
    script_path = project_root / "llm_verifier" / "llm_verifier.ts"
    issues_path = run_dir / f"{run_name}.json"

    if not script_path.exists():
        print(f"llm_verifier script not found at {script_path}")
        return 1

    if not issues_path.exists():
        print(f"Normalized issues JSON not found: {issues_path}")
        return 1

    thread_count = max(1, threads)
    npx_binary = resolve_npx_binary()
    cmd = [
        npx_binary,
        "ts-node",
        "--transpile-only",
        str(script_path),
        "--repo",
        str(repo_root),
        "--issues",
        str(issues_path),
        "--out-dir",
        str(run_dir),
        "--threads",
        str(thread_count),
    ]
    print(f"Running llm_verifier: {' '.join(cmd)} (cwd={project_root})")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def main() -> int:
    runner = prompt_runner()
    language = prompt_language()
    codeql_suite: str | None = None
    semgrep_configs: list[str] = []
    default_semgrep_target = Path(__file__).resolve().parent.parent
    semgrep_target: Path = default_semgrep_target
    semgrep_jobs: int | None = None
    semgrep_timeout: int | None = None
    semgrep_timeout_threshold: int | None = None
    semgrep_max_target_bytes: int | None = None

    if runner == "codeql":
        codeql_suite = prompt_codeql_suite(language)
    elif runner == "semgrep":
        semgrep_configs = prompt_semgrep_configs()
        semgrep_target = prompt_semgrep_target(default_semgrep_target)
        semgrep_jobs = prompt_optional_int("Semgrep jobs (-j) (blank to use runner default): ")
        semgrep_timeout = prompt_optional_int("Semgrep timeout seconds (blank to use runner default): ")
        semgrep_timeout_threshold = prompt_optional_int(
            "Semgrep timeout-threshold (blank to use runner default): "
        )
        semgrep_max_target_bytes = prompt_optional_int(
            "Semgrep max-target-bytes (blank to use runner default): "
        )
    threads = prompt_threads()

    oss = Path.cwd().parent.name
    run_dir, run_name, index = allocate_run_dir(oss, runner, language)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {run_dir}")
    print(f"Base name: {run_name} (for SARIF/JSON artifacts)")

    sarif_path = run_dir / f"{run_name}.sarif"

    if runner == "codeql":
        rc = run_codeql(language, run_dir, run_name, codeql_suite)
    elif runner == "semgrep":
        rc = run_semgrep(
            configs=semgrep_configs,
            target=semgrep_target,
            run_dir=run_dir,
            run_name=run_name,
            jobs=semgrep_jobs,
            timeout=semgrep_timeout,
            timeout_threshold=semgrep_timeout_threshold,
            max_target_bytes=semgrep_max_target_bytes,
        )
    else:
        print(f"Unknown runner: {runner}")
        return 1

    if rc != 0:
        return rc

    if not sarif_path.exists():
        print(f"SARIF file not found: {sarif_path}")
        return 1

    rc = run_generate_json(oss, runner, language, index, run_dir)
    if rc != 0:
        return rc

    return run_llm_verifier(run_dir, run_name, threads)


if __name__ == "__main__":
    sys.exit(main())
