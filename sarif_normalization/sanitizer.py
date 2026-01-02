import re
from typing import Iterable, List, Optional

from .sarif_types import NormalizedIssue


TEST_REGEX = re.compile(r"(?<![a-z])tests?(?![a-z])")
EXAMPLE_REGEX = re.compile(r"(?<![a-z])examples?(?![a-z])")
SPEC_REGEX = re.compile(r"(?<![a-z])specs?(?![a-z])")
FAVICON_REGEX = re.compile(r"(?<![a-z])favicon(?![a-z])")
DUMMY_REGEX = re.compile(r"(?<![a-z])dummy(?![a-z])")
MIGRATION_REGEX = re.compile(r"(?<![a-z])migrations?(?![a-z])")
ALEMBIC_REGEX = re.compile(r"(?<![a-z])alembic(?![a-z])")

BUNDLE_ARTIFACT_REGEX = re.compile(r"bundle[._-][a-z0-9]{5,}", re.IGNORECASE)
BUNDLE_EXTENSIONS = {".js", ".css", ".map"}

DROP_EXTENSIONS = {".json", ".yaml", ".yml", ".csv", ".md", ".txt", ".lock"}

SEGMENT_KEYWORDS = {
    "docker-compose",
    ".github",
    "font",
    "icon",
    "color",
    "dist",
    "_nuxt",
}

SEMGREP_DROP_PREFIXES = [
    "html.security.audit.",
    "generic.secrets.",
    "problem-based-packs.",
    "yaml.docker-compose.",
]
SEMGREP_DROP_EXACT = {
    "detect-non-literal-regexp",
    "incomplete-sanitization",
    "sha224-hash",
    "detect-insecure-websocket",
    "force-ssl-false",
    "generic.unicode.security.bidi.contains-bidirectional-characters",
    "hardcoded-jwt-secret",
    "insecure-hash-algorithm-sha1",
    "insecure-document-method",
    "insecure-object-assign",
    "missing-user",
    "missing-csrf-protection",
    "model-attributes-attr-accessible",
    "ssl-mode-no-verify",
    "weak-hashes-md5",
    "weak-hashes-sha1",
    "wildcard-postmessage-configuration",
}

CODEQL_DROP_EXACT = {
    "automatic-semicolon-insertion",
    "comparison-between-incompatible-types",
    "duplicate-property",
    "incomplete-url-substring-sanitization",
    "incomplete-multi-character-sanitization",
    "insufficient-password-hash",
    "ineffective-parameter-type",
    "incomplete-hostname-regexp",
    "incomplete-sanitization",
    "missing-await",
    "redundant-operation",
    "regex/duplicate-in-character-class",
    "superfluous-trailing-arguments",
    "template-syntax-in-string-literal",
    "unused-local-variable",
    "useless-assignment-to-local",
    "useless-comparison-test",
    "unneeded-defensive-code",
    "unsafe-jquery-plugin",
    "weak-sensitive-data-hasing",
    "weak-cryptographic-algorithm",
}


def sanitize_issues(issues: Iterable[NormalizedIssue]) -> List[NormalizedIssue]:
    return [issue for issue in issues if not should_drop_issue(issue)]


def should_drop_issue(issue: NormalizedIssue) -> bool:
    if _matches_rule_drop(issue):
        return True

    uris = _extract_issue_uris(issue)
    for uri in uris:
        if _matches_path_drop(uri):
            return True

    return False


def _matches_rule_drop(issue: NormalizedIssue) -> bool:
    tool_name = str(issue.get("tool", {}).get("name") or "").lower()
    rule_id = str(issue.get("rule", {}).get("id") or "").lower()
    if not tool_name or not rule_id:
        return False

    if "semgrep" in tool_name:
        if _matches_semgrep_rule(rule_id):
            return True
        for prefix in SEMGREP_DROP_PREFIXES:
            if rule_id.startswith(prefix):
                return True
        return False

    if "codeql" in tool_name:
        return _matches_codeql_rule(rule_id)

    return False


def _matches_semgrep_rule(rule_id: str) -> bool:
    if rule_id in SEMGREP_DROP_EXACT:
        return True
    tokens = [t for t in rule_id.split(".") if t]
    if not tokens:
        return False
    if tokens[-1] in SEMGREP_DROP_EXACT:
        return True
    if len(tokens) >= 2 and tokens[-2] in SEMGREP_DROP_EXACT:
        return True
    return False


def _matches_codeql_rule(rule_id: str) -> bool:
    if rule_id in CODEQL_DROP_EXACT:
        return True
    if "/" in rule_id:
        parts = [p for p in rule_id.split("/") if p]
        for part in parts:
            if part in CODEQL_DROP_EXACT:
                return True
        for drop in CODEQL_DROP_EXACT:
            if "/" in drop and (f"/{drop}" in rule_id or rule_id.endswith(drop)):
                return True
    return False


def _matches_path_drop(uri: str) -> bool:
    if not uri:
        return False

    path = _normalize_uri(uri)
    if not path:
        return False

    lowered = path.lower()
    filename = lowered.rsplit("/", 1)[-1]
    if filename == "dockerfile" or filename.endswith(".dockerfile"):
        return True

    ext = _file_extension(filename)
    if ext in DROP_EXTENSIONS:
        return True

    if ext in BUNDLE_EXTENSIONS:
        stem = filename[: -len(ext)]
        if BUNDLE_ARTIFACT_REGEX.search(stem):
            return True

    if _matches_segment_keywords(lowered):
        return True

    return False


def _matches_segment_keywords(path: str) -> bool:
    segments = [seg for seg in path.split("/") if seg]
    for seg in segments:
        if _segment_regex_match(seg):
            return True
        for keyword in SEGMENT_KEYWORDS:
            if keyword in seg:
                return True
    return False


def _segment_regex_match(segment: str) -> bool:
    if TEST_REGEX.search(segment):
        return True
    if EXAMPLE_REGEX.search(segment):
        return True
    if SPEC_REGEX.search(segment):
        return True
    if FAVICON_REGEX.search(segment):
        return True
    if DUMMY_REGEX.search(segment):
        return True
    if MIGRATION_REGEX.search(segment):
        return True
    if ALEMBIC_REGEX.search(segment):
        return True
    return False


def _extract_issue_uris(issue: NormalizedIssue) -> List[str]:
    uris: List[str] = []
    locations = issue.get("locations", {})
    primary = locations.get("primary", {}) if isinstance(locations, dict) else {}
    if isinstance(primary, dict):
        uri = primary.get("uri")
        if uri:
            uris.append(str(uri))

    related = locations.get("related") if isinstance(locations, dict) else None
    if isinstance(related, list):
        for entry in related:
            if isinstance(entry, dict):
                uri = entry.get("uri")
                if uri:
                    uris.append(str(uri))

    return uris


def _normalize_uri(uri: str) -> Optional[str]:
    text = uri.strip()
    if not text:
        return None

    text = text.replace("\\", "/")
    if text.startswith("file://"):
        text = text[len("file://") :]
        if text.startswith("/"):
            text = text[1:]
    return text


def _file_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1]
