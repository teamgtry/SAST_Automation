import argparse
import json
from pathlib import Path

from sarif_normalization.parser import parse_sarif
from sarif_normalization.sanitizer import sanitize_issues


def build_sarif_filename(oss: str, sast: str, lang: str, index: str) -> Path:
    """Compose SARIF filename oss-sast-lang-index.sarif in current directory."""
    return Path(f"{oss}-{sast}-{lang}-{index}.sarif")


def build_output_path(oss: str, sast: str, lang: str, index: str) -> Path:
    """Compose JSON output path ../oss-sast-lang-index/oss-sast-lang-index.json."""
    base_name = f"{oss}-{sast}-{lang}-{index}"
    return Path("..") / base_name / f"{base_name}.json"


def build_sanitized_output_path(oss: str, sast: str, lang: str, index: str) -> Path:
    """Compose sanitized JSON output path ../oss-sast-lang-index/oss-sast-lang-index.sanitized.json."""
    base_name = f"{oss}-{sast}-{lang}-{index}"
    return Path("..") / base_name / f"{base_name}.sanitized.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a SARIF file named <oss>-<sast>-<lang>-<index>.sarif and export normalized issues."
    )
    parser.add_argument("--oss", required=True, help="OSS/project name component.")
    parser.add_argument("--sast", required=True, help="SAST tool name component (e.g., codeql).")
    parser.add_argument("--lang", required=True, help="Language component (e.g., javascript).")
    parser.add_argument("--index", required=True, help="Index component (e.g., 0, 1...).")
    args = parser.parse_args()

    sarif_path = build_sarif_filename(args.oss, args.sast, args.lang, args.index)
    output_path = build_output_path(args.oss, args.sast, args.lang, args.index)
    sanitized_output_path = build_sanitized_output_path(
        args.oss, args.sast, args.lang, args.index
    )

    with sarif_path.open("r", encoding="utf-8") as handle:
        sarif_data = json.load(handle)

    issues = parse_sarif(sarif_data)
    sanitized_issues = sanitize_issues(issues)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(issues, handle, indent=2, ensure_ascii=False)

    with sanitized_output_path.open("w", encoding="utf-8") as handle:
        json.dump(sanitized_issues, handle, indent=2, ensure_ascii=False)

    print(f"Exported {len(issues)} issues to {output_path}")
    print(f"Exported {len(sanitized_issues)} issues to {sanitized_output_path}")


if __name__ == "__main__":
    main()
