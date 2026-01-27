from __future__ import annotations

import os
import re
from typing import Dict, List, Optional


POST_DROP_FILENAME_KEYWORDS = {
    "test-related": [
        "test",
        "spec",
        "e2e",
        "mock",
        "fixture",
        "stub",
        "snapshot",
    ],
    "doc-example": [
        "doc",
        "documentation",
        "example",
        "sample",
        "demo",
        "tutorial",
    ],
    "migration": [
        "migration",
        "migrate",
        "alembic",
        "schema",
        "flyway",
        "liquibase",
        "backup",
    ],
    "generated": [
        "proto",
        "protobuf",
        "swagger",
        "generated",
        "__generated__",
    ],
    "i18n": [
        "locale",
        "i18n",
        "l10n",
        "lang",
        "translation",
    ],
}


def _compile_filename_patterns() -> Dict[str, List[re.Pattern[str]]]:
    compiled: Dict[str, List[re.Pattern[str]]] = {}
    for reason, keywords in POST_DROP_FILENAME_KEYWORDS.items():
        patterns: List[re.Pattern[str]] = []
        for kw in keywords:
            # Allow simple plural forms like "alembics" while keeping boundary safety.
            suffix = "" if kw.endswith("s") else "s?"
            patterns.append(
                re.compile(
                    rf"(^|[_\-.]){re.escape(kw)}{suffix}([_\-.]|$)", re.IGNORECASE
                )
            )
        compiled[reason] = patterns
    return compiled


POST_DROP_FILENAME_PATTERNS = _compile_filename_patterns()


POST_DROP_EXTENSIONS = {
    "compiled-artifact": [
        ".min.js",
        ".min.css",
        ".bundle.js",
        ".chunk.js",
        ".map",
        ".pyc",
        ".pyo",
        ".pyd",
        ".class",
        ".o",
        ".so",
        ".dll",
        ".exe",
        ".dylib",
        ".a",
        ".lib",
    ],
    "documentation": [
        ".md",
        ".markdown",
        ".rst",
        ".txt",
        ".adoc",
        ".asciidoc",
    ],
    "image": [
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".webp",
        ".bmp",
        ".tiff",
        ".avif",
    ],
    "font": [
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".otf",
    ],
    "media": [
        ".mp4",
        ".webm",
        ".avi",
        ".mov",
        ".mkv",
        ".flv",
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".aac",
        ".m4a",
    ],
    "archive": [
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".rar",
        ".7z",
    ],
    "temp-log": [
        ".log",
        ".swp",
        ".swo",
    ],
    "config-data": [
        ".json",
        ".yaml",
        ".yml",
        ".lock",
        ".sarif",
    ],
}


def should_post_drop(code_unit: Dict) -> Optional[str]:
    path = code_unit.get("file", "").lower()
    filename = os.path.basename(path)

    for reason, patterns in POST_DROP_FILENAME_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(filename):
                return f"filename:{reason}"

    for reason, extensions in POST_DROP_EXTENSIONS.items():
        for ext in extensions:
            if filename.endswith(ext):
                return f"extension:{reason}"

    return None
