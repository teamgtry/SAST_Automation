#!/usr/bin/env python3
"""
Repo RAG Loader/Retriever (Hybrid: BM25 + Embeddings)

Usage (from your llm_verifier):
  from repo_rag import RepoRAG
  rag = RepoRAG.load("index_dir")
  hits = rag.search("sanitizeHtml allowlist DOMPurify", top_k=8, path_hint="src/ui")
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def cosine_sim_matrix(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # vectors are assumed normalized; if not, normalize here
    q = query_vec.astype(np.float32)
    if q.ndim == 1:
        q = q[None, :]
    # mat: (N, D), q: (1, D)
    return (mat @ q.T).squeeze(-1)  # (N,)


@dataclass
class Hit:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    language: str
    symbols: List[str]
    snippet: str
    score_bm25: float
    score_vec: float
    score_final: float


class BM25Index:
    """
    Minimal BM25 implementation (same as repo_rag_build.py) for unpickling.
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


class RepoRAG:
    def __init__(self, chunks: List[dict], meta: List[dict], bm25_obj, tokenized, embeddings: Optional[np.ndarray]):
        self.chunks = chunks
        self.meta = meta
        self.bm25 = bm25_obj
        self.tokenized = tokenized
        self.embeddings = embeddings

        # build quick map chunk_id -> chunk text
        self.text_by_id: Dict[str, str] = {c["chunk_id"]: c["text"] for c in chunks}

    @staticmethod
    def load(index_dir: str) -> "RepoRAG":
        d = Path(index_dir)

        # load chunks.jsonl
        chunks = []
        with (d / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

        # load bm25
        # Ensure pickled BM25Index (often saved from __main__) can be resolved.
        # When repo_rag_build.py is run as a script, the class is pickled as __main__.BM25Index.
        # During load (e.g., from llm_verifier.py), __main__ is different, so we patch it.
        import __main__  # type: ignore
        if not hasattr(__main__, "BM25Index"):
            __main__.BM25Index = BM25Index
        with (d / "bm25.pkl").open("rb") as f:
            obj = pickle.load(f)
        bm25 = obj["bm25"]
        tokenized = obj["tokenized"]

        # load meta
        with (d / "meta.pkl").open("rb") as f:
            meta = pickle.load(f)

        # load embeddings optional
        emb_path = d / "embeddings.npy"
        embeddings = np.load(emb_path).astype(np.float32) if emb_path.exists() else None

        return RepoRAG(chunks=chunks, meta=meta, bm25_obj=bm25, tokenized=tokenized, embeddings=embeddings)

    def _embed_query(self, query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        """
        Query embedding should use the same model as build-time.
        If you use embeddings, keep model_name consistent across build/search.
        """
        if self.embeddings is None:
            raise RuntimeError("No embeddings loaded. Rebuild index without --no-embeddings.")

        cache_dir = Path(__file__).resolve().parent.parent / ".hf_cache"
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers") from e

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                model = SentenceTransformer(model_name)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(2.0 * (2 ** attempt))
                    continue
                raise
        q = model.encode([query], normalize_embeddings=True)
        return np.asarray(q[0], dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 8,
        alpha: float = 0.55,
        path_hint: Optional[str] = None,
        embed_model: str = "all-MiniLM-L6-v2",
    ) -> List[Hit]:
        """
        Hybrid scoring:
          final = alpha * norm(bm25) + (1-alpha) * norm(vec)
        If embeddings missing: BM25-only.

        path_hint:
          If provided, boosts chunks whose path contains the hint.
        """
        q_tokens = tokenize(query)
        bm25_scores = self.bm25.scores(q_tokens).astype(np.float32)

        # normalize bm25 to 0..1
        if bm25_scores.size:
            bm25_min, bm25_max = float(bm25_scores.min()), float(bm25_scores.max())
            bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-8)
        else:
            bm25_norm = bm25_scores

        if self.embeddings is not None:
            qv = self._embed_query(query, model_name=embed_model)
            vec_scores = cosine_sim_matrix(qv, self.embeddings).astype(np.float32)
            # normalize vec to 0..1 (cosine can be -1..1)
            vec_min, vec_max = float(vec_scores.min()), float(vec_scores.max())
            vec_norm = (vec_scores - vec_min) / (vec_max - vec_min + 1e-8)
        else:
            vec_scores = np.zeros_like(bm25_scores, dtype=np.float32)
            vec_norm = vec_scores

        final = alpha * bm25_norm + (1.0 - alpha) * vec_norm

        # optional path boost
        if path_hint:
            hint = path_hint.replace("\\", "/")
            for i, m in enumerate(self.meta):
                if hint in m["path"]:
                    final[i] += 0.10  # small boost

        # pick top_k
        idx = np.argsort(-final)[:top_k]
        hits: List[Hit] = []
        for i in idx:
            m = self.meta[i]
            chunk_id = m["chunk_id"]
            snippet = self.text_by_id.get(chunk_id, "")

            hits.append(Hit(
                chunk_id=chunk_id,
                path=m["path"],
                start_line=m["start_line"],
                end_line=m["end_line"],
                language=m["language"],
                symbols=m["symbols"],
                snippet=snippet,
                score_bm25=float(bm25_scores[i]) if bm25_scores.size else 0.0,
                score_vec=float(vec_scores[i]) if self.embeddings is not None else 0.0,
                score_final=float(final[i]),
            ))
        return hits
