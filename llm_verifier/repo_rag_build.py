#!/usr/bin/env python3
"""
Offline Repo RAG Index Builder (Hybrid: BM25 + Embeddings)

- Chunking:
  * Python: function/class chunks via AST (best effort)
  * Others: fallback to fixed line windows (size/overlap)
- Index:
  * BM25 over tokenized chunks
  * Embeddings via sentence-transformers (recommended)
- Metadata per chunk:
  path, start_line, end_line, symbols, language, sha1

Output directory layout:
  index_dir/
    chunks.jsonl
    bm25.pkl
    embeddings.npy
    meta.pkl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------
# Utilities
# ---------------------------

CODE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".cs", ".php", ".rb", ".rs", ".kt", ".swift", ".scala"
}

BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".jar", ".class", ".exe", ".dll", ".so", ".dylib"}

IGNORE_DIRS = {".git", "node_modules", "dist", "build", "out", ".next", ".venv", "venv", "__pycache__"}

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")

SYMBOL_RE_LIST = [
    re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"\bexport\s+function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"\bconst\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\("),
    re.compile(r"\blet\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\("),
]


def infer_language(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".h": "c/cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".php": "php",
        ".rb": "ruby",
        ".rs": "rust",
        ".kt": "kotlin",
        ".swift": "swift",
        ".scala": "scala",
    }.get(ext, "unknown")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def extract_symbols_heuristic(text: str, max_symbols: int = 20) -> List[str]:
    found: List[str] = []
    for rx in SYMBOL_RE_LIST:
        for m in rx.finditer(text):
            found.append(m.group(1))
            if len(found) >= max_symbols:
                return found
    # de-dup preserve order
    out = []
    seen = set()
    for s in found:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:max_symbols]


# ---------------------------
# Chunk model
# ---------------------------

@dataclass
class Chunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    language: str
    symbols: List[str]
    text: str
    sha1: str


# ---------------------------
# Chunking: Python AST
# ---------------------------

def chunk_python_ast(path: Path, text: str) -> List[Chunk]:
    """
    Extract function/class blocks using Python AST (best-effort).
    Falls back to empty list if parsing fails or lineno info missing.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    chunks: List[Chunk] = []

    # For Python 3.8+, nodes have end_lineno
    def make_chunk(node: ast.AST, sym_name: str) -> Optional[Chunk]:
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        if lineno is None or end_lineno is None:
            return None
        start = max(1, int(lineno))
        end = max(start, int(end_lineno))
        body = "\n".join(lines[start - 1:end])
        if not body.strip():
            return None

        lang = infer_language(path)
        symbols = [sym_name]
        # also pull nested symbols (heuristic) inside the block
        symbols += [s for s in extract_symbols_heuristic(body) if s != sym_name]
        symbols = symbols[:20]

        chunk_id = f"{path.as_posix()}:{start}-{end}:{sha1_text(body)[:10]}"
        return Chunk(
            chunk_id=chunk_id,
            path=path.as_posix(),
            start_line=start,
            end_line=end,
            language=lang,
            symbols=symbols,
            text=body,
            sha1=sha1_text(body),
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = getattr(node, "name", None) or "anonymous"
            c = make_chunk(node, name)
            if c:
                chunks.append(c)

    return chunks


# ---------------------------
# Chunking: Fallback line windows
# ---------------------------

def chunk_by_lines(path: Path, text: str, window: int = 220, overlap: int = 60) -> List[Chunk]:
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []

    lang = infer_language(path)
    chunks: List[Chunk] = []
    step = max(1, window - overlap)
    start = 1

    while start <= n:
        end = min(n, start + window - 1)
        body = "\n".join(lines[start - 1:end]).rstrip()
        if body.strip():
            symbols = extract_symbols_heuristic(body)
            chunk_id = f"{path.as_posix()}:{start}-{end}:{sha1_text(body)[:10]}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                path=path.as_posix(),
                start_line=start,
                end_line=end,
                language=lang,
                symbols=symbols,
                text=body,
                sha1=sha1_text(body),
            ))
        start += step

    return chunks


# ---------------------------
# Repo walk
# ---------------------------

def iter_repo_files(repo_root: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo_root):
        # prune ignore dirs
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fn in files:
            p = Path(root) / fn
            ext = p.suffix.lower()
            if ext in BINARY_EXTS:
                continue
            if ext not in CODE_EXTS:
                continue
            yield p


def read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


# ---------------------------
# BM25 (minimal implementation)
# ---------------------------

class BM25Index:
    """
    Minimal BM25 implementation to avoid hard dependency on rank_bm25.
    """

    def __init__(self, tokenized_corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.N = len(tokenized_corpus)
        self.doc_lens = np.array([len(d) for d in tokenized_corpus], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if self.N else 0.0

        # df
        df: Dict[str, int] = {}
        for doc in tokenized_corpus:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        self.df = df
        # idf
        self.idf: Dict[str, float] = {}
        for t, f in df.items():
            # BM25+ style idf
            self.idf[t] = float(np.log(1 + (self.N - f + 0.5) / (f + 0.5)))

    def scores(self, query_tokens: List[str]) -> np.ndarray:
        if self.N == 0:
            return np.array([], dtype=np.float32)

        qfreq: Dict[str, int] = {}
        for t in query_tokens:
            qfreq[t] = qfreq.get(t, 0) + 1

        scores = np.zeros(self.N, dtype=np.float32)
        for i, doc in enumerate(self.corpus):
            if not doc:
                continue
            dl = self.doc_lens[i]
            # tf
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1

            s = 0.0
            for t, _ in qfreq.items():
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-8)))
                s += idf * (freq * (self.k1 + 1) / (denom + 1e-8))
            scores[i] = s

        return scores


# ---------------------------
# Embeddings
# ---------------------------

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """
    Uses sentence-transformers if available. Install:
      pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install with:\n"
            "  pip install sentence-transformers\n"
            "Or run with --no-embeddings to build BM25-only index."
        ) from e

    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


# ---------------------------
# Build index
# ---------------------------

def build_chunks(repo_root: Path, window: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for fp in iter_repo_files(repo_root):
        text = read_text_file(fp)
        if not text or not text.strip():
            continue

        file_chunks: List[Chunk] = []
        if fp.suffix.lower() == ".py":
            file_chunks = chunk_python_ast(fp, text)

        # fallback if AST produced nothing or for non-python
        if not file_chunks:
            file_chunks = chunk_by_lines(fp, text, window=window, overlap=overlap)

        chunks.extend(file_chunks)

    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to repo root")
    ap.add_argument("--out", required=True, help="Output index directory")
    ap.add_argument("--window", type=int, default=220, help="Line window size for fallback chunking")
    ap.add_argument("--overlap", type=int, default=60, help="Line overlap for fallback chunking")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    ap.add_argument("--no-embeddings", action="store_true", help="Build BM25-only (no embeddings.npy)")
    args = ap.parse_args()

    repo_root = Path(args.repo).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Building chunks from: {repo_root}")
    chunks = build_chunks(repo_root, window=args.window, overlap=args.overlap)
    print(f"[+] Total chunks: {len(chunks)}")

    # Save chunks.jsonl (text + metadata)
    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            row = asdict(c)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[+] Wrote: {chunks_path}")

    # Build BM25
    print("[+] Building BM25 index...")
    tokenized = [tokenize(c.text) for c in chunks]
    bm25 = BM25Index(tokenized)
    bm25_path = out_dir / "bm25.pkl"
    with bm25_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    print(f"[+] Wrote: {bm25_path}")

    # Embeddings
    embeddings = None
    if not args.no_embeddings:
        print("[+] Computing embeddings...")
        embeddings = embed_texts([c.text for c in chunks], model_name=args.embed_model)
        emb_path = out_dir / "embeddings.npy"
        np.save(emb_path, embeddings)
        print(f"[+] Wrote: {emb_path}")
    else:
        print("[!] Skipped embeddings (BM25-only).")

    # Meta (lightweight)
    meta = [{
        "chunk_id": c.chunk_id,
        "path": c.path,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "language": c.language,
        "symbols": c.symbols,
        "sha1": c.sha1,
    } for c in chunks]
    meta_path = out_dir / "meta.pkl"
    with meta_path.open("wb") as f:
        pickle.dump(meta, f)
    print(f"[+] Wrote: {meta_path}")

    print("[+] Done.")


if __name__ == "__main__":
    main()
