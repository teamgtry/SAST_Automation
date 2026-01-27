from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

def prompt_language() -> str:
    while True:
        lang = input("Language (e.g., javascript, python): ").strip()
        if lang:
            return lang
        print("Language is required.")

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


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        raw = input(f"{prompt}{suffix}").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def allocate_run_dir(oss: str, language: str) -> tuple[Path, str, int]:
    """
    Find the first available ../<oss>-<runner>-<lang>-<index> directory.
    Returns (run_dir_path, base_name_with_index, index).
    """
    parent = Path.cwd()
    base_prefix = f"{oss}-semgrep-{language}"
    index = 0
    while True:
        base_name = f"{base_prefix}-{index}"
        candidate = parent / base_name
        if not candidate.exists():
            return candidate, base_name, index
        index += 1

def run_semgrep(
    configs: list[str],
    target: Path,
    run_dir: Path,
    run_name: str,
    pro: bool = False,
    jobs: int | None = None,
    timeout: int | None = None,
    timeout_threshold: int | None = None,
    max_target_bytes: int | None = None,
    quiet: bool = True,
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
    if pro:
        cmd.append("--pro")
    if quiet:
        cmd.append("--quiet")
    if jobs is not None:
        cmd.extend(["--jobs", str(jobs)])
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])
    if timeout_threshold is not None:
        cmd.extend(["--timeout-threshold", str(timeout_threshold)])
    if max_target_bytes is not None:
        cmd.extend(["--max-target-bytes", str(max_target_bytes)])

    if not quiet:
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
        if not quiet:
            print(f"SARIF saved to: {sarif_path}")
    else:
        if not quiet:
            print(f"SARIF file not found at expected location: {runner_default_sarif}")
    return result.returncode


def run_generate_json(
    oss: str,
    runner: str,
    language: str,
    index: int,
    run_dir: Path,
    source_root: Path,
    quiet: bool = True,
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
        "--source-root",
        str(source_root),
    ]
    if quiet:
        cmd.append("--quiet")
    else:
        print(f"Generating normalized JSON: {' '.join(cmd)} (cwd={run_dir})")
    result = subprocess.run(cmd, cwd=run_dir, env=env)
    return result.returncode


def _rag_index_ready(index_dir: Path) -> bool:
    required = [
        index_dir / "chunks.jsonl",
        index_dir / "bm25.pkl",
        index_dir / "meta.pkl",
    ]
    return all(p.exists() for p in required)


def run_repo_rag_build(repo_root: Path, index_dir: Path, run_dir: Path, quiet: bool = True) -> int:
    script_path = Path(__file__).resolve().parent / "llm_verifier" / "repo_rag_build.py"
    if not script_path.exists():
        print(f"repo_rag_build not found at {script_path}")
        return 1

    if _rag_index_ready(index_dir):
        if not quiet:
            print(f"Repo RAG index already exists: {index_dir}")
        return 0

    cmd = [
        sys.executable,
        str(script_path),
        "--repo",
        str(repo_root),
        "--out",
        str(index_dir),
    ]
    if not quiet:
        print(f"Building Repo RAG index: {' '.join(cmd)} (cwd={run_dir})")
    result = subprocess.run(cmd, cwd=run_dir)
    return result.returncode


def run_llm_verifier(
    repo_root: Path,
    issues_path: Path,
    index_dir: Path,
    out_dir: Path,
    run_dir: Path,
    quiet: bool = True,
) -> int:
    script_path = Path(__file__).resolve().parent / "llm_verifier" / "llm_verifier.py"
    if not script_path.exists():
        print(f"llm_verifier not found at {script_path}")
        return 1

    cmd = [
        sys.executable,
        str(script_path),
        "--repo",
        str(repo_root),
        "--issues",
        str(issues_path),
        "--index-dir",
        str(index_dir),
        "--out-dir",
        str(out_dir),
    ]
    if not quiet:
        print(f"Running LLM verifier: {' '.join(cmd)} (cwd={run_dir})")
    result = subprocess.run(cmd, cwd=run_dir)
    return result.returncode


def main() -> int:
    runner = "semgrep"
    language = prompt_language()
    semgrep_configs: list[str] = []
    default_semgrep_target = Path(__file__).resolve().parent.parent
    semgrep_target: Path = default_semgrep_target
    semgrep_jobs: int | None = None
    semgrep_timeout: int | None = None
    semgrep_timeout_threshold: int | None = None
    semgrep_max_target_bytes: int | None = None
    semgrep_pro: bool = False


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
    semgrep_pro = prompt_yes_no("Use Semgrep Pro engine?", default=False)
    oss = Path.cwd().parent.name
    run_dir, run_name, index = allocate_run_dir(oss, language)
    run_dir.mkdir(parents=True, exist_ok=True)

    sarif_path = run_dir / f"{run_name}.sarif"

    rc = run_semgrep(
        configs=semgrep_configs,
        target=semgrep_target,
        run_dir=run_dir,
        run_name=run_name,
        pro=semgrep_pro,
        jobs=semgrep_jobs,
        timeout=semgrep_timeout,
        timeout_threshold=semgrep_timeout_threshold,
        max_target_bytes=semgrep_max_target_bytes,
    )

    if rc != 0:
        return rc

    if not sarif_path.exists():
        print(f"SARIF file not found: {sarif_path}")
        return 1

    rc = run_generate_json(
        oss, runner, language, index, run_dir, semgrep_target, quiet=True
    )
    if rc != 0:
        return rc

    repo_root = semgrep_target if semgrep_target.is_dir() else semgrep_target.parent
    rag_index_dir = repo_root / ".rag_index"
    rc = run_repo_rag_build(repo_root, rag_index_dir, run_dir, quiet=True)
    if rc != 0:
        return rc

    issues_path = run_dir / f"{run_name}.sanitized.json"
    llm_out_dir = run_dir / "llm_verifier"
    rc = run_llm_verifier(
        repo_root=repo_root,
        issues_path=issues_path,
        index_dir=rag_index_dir,
        out_dir=llm_out_dir,
        run_dir=run_dir,
        quiet=True,
    )
    if rc != 0:
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
