from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - runtime dependency
    END = None
    StateGraph = None


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


class DraftDecision(TypedDict):
    decision: str
    reason: str


class Question(TypedDict):
    question: str
    needed_evidence: str


class Answer(TypedDict):
    question: str
    answer: str
    evidence: List[Dict[str, str]]


class FinalDecision(TypedDict):
    decision: str
    drop_confidence: float
    fp_tags: List[str]
    evidence: List[Dict[str, str]]


class CaseState(TypedDict, total=False):
    case_id: str
    rule_id: str
    file_path: str
    code_snippet: str
    issue: Dict[str, Any]
    draft: DraftDecision
    questions: List[Question]
    queries: List[Dict[str, Optional[str]]]
    retrieved_context: List[Dict[str, Any]]
    answers: List[Answer]
    final: FinalDecision
    logger: "CaseLogger"
    cache: "QACache"


class LLMClient:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class LangChainGeminiClient(LLMClient):
    def __init__(self, model: str, temperature: float, system_prompt: Optional[str] = None) -> None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
            from langchain_core.messages import HumanMessage  # type: ignore
            from langchain_core.messages import SystemMessage  # type: ignore
        except ImportError as exc:
            raise RuntimeError("langchain_google_genai or langchain_core not installed") from exc
        self._HumanMessage = HumanMessage
        self._SystemMessage = SystemMessage
        self._client = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            response_mime_type="application/json",
        )
        self._system_prompt = system_prompt

    def complete(self, prompt: str) -> str:
        messages = []
        if self._system_prompt:
            messages.append(self._SystemMessage(content=self._system_prompt))
        messages.append(self._HumanMessage(content=prompt))
        last_exc: Optional[BaseException] = None
        for attempt in range(3):
            try:
                response = self._client.invoke(messages)
                return getattr(response, "content", "") or ""
            except Exception as exc:  # pragma: no cover - runtime dependency
                last_exc = exc
                if not is_overloaded_error(exc):
                    raise
                if attempt < 2:
                    time.sleep(1.5 * (2 ** attempt))
                    continue
                raise OverloadedError("Gemini model overloaded after retries") from exc
        raise OverloadedError("Gemini model overloaded after retries") from last_exc


class MockLLMClient(LLMClient):
    def complete(self, prompt: str) -> str:
        if "\"draft\"" in prompt or "draft_fp_filter" in prompt:
            return json.dumps({"decision": "keep", "reason": "mock: insufficient evidence"})
        if "\"questions\"" in prompt or "cove_generate_questions" in prompt:
            return json.dumps({"questions": []})
        if "\"queries\"" in prompt or "rag_query_planner" in prompt:
            return json.dumps({"queries": []})
        if "\"answers\"" in prompt or "cove_answer_questions" in prompt:
            return json.dumps({"answers": []})
        if "\"final\"" in prompt or "refine_decision" in prompt:
            return json.dumps({
                "decision": "keep",
                "drop_confidence": 0.0,
                "fp_tags": [],
                "evidence": [],
            })
        return "{}"


class CaseLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._txt_path = path.with_suffix(".txt")
        self._lock = threading.Lock()

    def log(self, node: str, prompt: str, response: str) -> None:
        entry = {"node": node, "prompt": prompt, "response": response}
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")
            with self._txt_path.open("a", encoding="utf-8") as f:
                f.write(f"[{node}] response\n{response}\n\n")


class QACache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            try:
                self._cache = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = value
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._cache, ensure_ascii=True, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Verifier (LangGraph)")
    parser.add_argument("--repo", required=True, help="Repo root dir")
    parser.add_argument("--issues", required=True, help="Normalized issues JSON")
    parser.add_argument("--index-dir", required=True, help="Repo RAG index dir")
    parser.add_argument("--out-dir", required=True, help="Output dir")
    parser.add_argument("--threads", type=int, default=1, help="Worker threads")
    parser.add_argument("--case-id", default=None, help="Process a single case id")
    parser.add_argument("--model", default="gemini-1.5-flash", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--top-k", type=int, default=8, help="RAG top-k")
    parser.add_argument("--alpha", type=float, default=0.55, help="BM25/vec mix")
    parser.add_argument("--confidence-threshold", type=float, default=0.88, help="Drop gate threshold")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM outputs")
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("[extract_json] JSON decode failed on full text.", file=sys.stderr)
    codeblock_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if codeblock_match:
        try:
            return json.loads(codeblock_match.group(1))
        except json.JSONDecodeError:
            print("[extract_json] JSON decode failed on codeblock.", file=sys.stderr)
            return {}
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            print("[extract_json] JSON decode failed on extracted block.", file=sys.stderr)
            return {}
    print("[extract_json] No JSON object found in text.", file=sys.stderr)
    return {}


def tokenize(text: str) -> List[str]:
    return [m.group(0) for m in TOKEN_RE.finditer(text)]


def normalize_uri(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return None
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    return uri


def parse_line_range(line_value: Any) -> Optional[tuple[int, int]]:
    if line_value is None:
        return None
    if isinstance(line_value, (list, tuple)) and len(line_value) >= 2:
        try:
            return int(line_value[0]), int(line_value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(line_value, int):
        return line_value, line_value
    if isinstance(line_value, str):
        line_value = line_value.strip()
        if "-" in line_value:
            parts = [p.strip() for p in line_value.split("-") if p.strip().isdigit()]
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        if "," in line_value:
            nums = [int(p.strip()) for p in line_value.split(",") if p.strip().isdigit()]
            if nums:
                return min(nums), max(nums)
        if line_value.isdigit():
            n = int(line_value)
            return n, n
    return None


def read_snippet(repo_root: Path, file_uri: Optional[str], line_value: Any) -> str:
    path_str = normalize_uri(file_uri)
    if not path_str:
        return ""
    file_path = (repo_root / path_str).resolve()
    if not file_path.exists():
        return ""
    line_range = parse_line_range(line_value)
    if not line_range:
        return ""
    start, end = line_range
    try:
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    start_idx = max(1, start)
    end_idx = min(len(lines), end)
    snippet = "\n".join(lines[start_idx - 1 : end_idx])
    return snippet


def normalize_sanitized_path(path_str: Optional[str], repo_root: Path) -> str:
    if not path_str:
        return ""
    if path_str.startswith("file://"):
        path_str = path_str[len("file://") :]
    path_norm = path_str.replace("\\", "/")
    repo_norm = str(repo_root).replace("\\", "/")
    if path_norm.lower().startswith(repo_norm.lower().rstrip("/") + "/"):
        rel = path_norm[len(repo_norm.rstrip("/")) + 1 :]
        return rel
    repo_name = repo_root.name
    marker = f"/{repo_name}/"
    idx = path_norm.lower().find(marker.lower())
    if idx >= 0:
        return path_norm[idx + len(marker) :]
    return path_norm


def build_code_snippet_from_units(repo_root: Path, units: List[Dict[str, Any]]) -> str:
    rendered: List[str] = []
    for unit in units:
        roles = ",".join(unit.get("roles") or [])
        file_path = normalize_sanitized_path(unit.get("file"), repo_root)
        line_range = unit.get("line_range")
        line_value = ""
        line_tuple = parse_line_range(line_range)
        if line_tuple:
            line_value = f"{line_tuple[0]}-{line_tuple[1]}"
        header = f"# role: {roles} file: {file_path} lines: {line_value}".strip()
        body = unit.get("full_code") or read_snippet(repo_root, unit.get("file"), line_range)
        if body:
            rendered.append(f"{header}\n{body}".strip())
    return "\n\n".join(rendered)


def build_case_input(repo_root: Path, issue: Dict[str, Any]) -> Dict[str, Any]:
    if "code_units" in issue and "rule_id" in issue:
        case_id = str(issue.get("issue_id", "unknown"))
        rule_id = str(issue.get("rule_id", "unknown"))
        units = issue.get("code_units") or []
        primary = units[0] if units else {}
        file_path = normalize_sanitized_path(primary.get("file"), repo_root)
        code_snippet = build_code_snippet_from_units(repo_root, units)
        if not code_snippet:
            code_snippet = read_snippet(repo_root, primary.get("file"), primary.get("line_range"))
        return {
            "case_id": case_id,
            "rule_id": rule_id,
            "file_path": file_path,
            "code_snippet": code_snippet,
            "issue": issue,
        }

    case_id = str(issue.get("issue_id", "unknown"))
    rule_id = str(issue.get("rule", {}).get("id", "unknown"))
    primary = issue.get("locations", {}).get("primary", {})
    uri = primary.get("uri")
    line_value = primary.get("region", {}).get("Line")
    code_snippet = read_snippet(repo_root, uri, line_value)
    file_path = normalize_uri(uri) or ""
    return {
        "case_id": case_id,
        "rule_id": rule_id,
        "file_path": file_path,
        "code_snippet": code_snippet,
        "issue": issue,
    }


def build_draft_prompt(state: CaseState) -> str:
    return (
        "Node: draft_fp_filter\n"
        "You are a pragmatic SAST verifier.\n"
        "Input fields:\n"
        f"- rule_id: {state.get('rule_id')}\n"
        f"- file: {state.get('file_path')}\n"
        f"- code_snippet:\n{state.get('code_snippet')}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "decision": "drop_candidate" | "keep",\n'
        '  "reason": "why this is a drop candidate; if unsure, lean drop_candidate when patterns indicate likely FP"\n'
        "}\n"
        "Do NOT assert drop; use drop_candidate when there are reasonable indicators of FP."
    )


def build_questions_prompt(state: CaseState) -> str:
    return (
        "Node: cove_generate_questions\n"
        "Generate questions required to confirm a false positive.\n"
        "Return JSON only:\n"
        "{\n"
        '  "questions": [\n'
        '    {"question": "...", "needed_evidence": "specific code evidence needed"}\n'
        "  ]\n"
        "}\n"
        f"rule_id: {state.get('rule_id')}\n"
        f"file: {state.get('file_path')}\n"
        f"code_snippet:\n{state.get('code_snippet')}\n"
    )


def build_query_plan_prompt(state: CaseState) -> str:
    questions = state.get("questions", [])
    return (
        "Node: rag_query_planner\n"
        "Convert questions into repo search queries.\n"
        "Prioritize:\n"
        "1) same file definitions/usages\n"
        "2) imported module definitions\n"
        "3) global search\n"
        "Return JSON only:\n"
        "{\n"
        '  "queries": [\n'
        '    {"query": "...", "path_hint": "optional path"}\n'
        "  ]\n"
        "}\n"
        f"file: {state.get('file_path')}\n"
        f"questions: {json.dumps(questions, ensure_ascii=True)}\n"
    )


def build_answer_prompt(question: Question, context: List[Dict[str, Any]]) -> str:
    return (
        "Node: cove_answer_questions\n"
        "Answer using retrieved context only. If line-based evidence is missing, answer 'unknown'.\n"
        "Return JSON only:\n"
        "{\n"
        '  "answers": [\n'
        '    {\n'
        '      "question": "...",\n'
        '      "answer": "answer or unknown",\n'
        '      "evidence": [\n'
        '        {"path": "...", "line": "start-end", "snippet": "..."}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        f"question: {question.get('question')}\n"
        f"needed_evidence: {question.get('needed_evidence')}\n"
        f"retrieved_context: {json.dumps(context, ensure_ascii=True)}\n"
    )


def build_refine_prompt(state: CaseState) -> str:
    return (
        "Node: refine_decision\n"
        "Revise draft decision using Q/A.\n"
        "Return JSON only:\n"
        "{\n"
        '  "decision": "drop_fp" | "keep",\n'
        '  "drop_confidence": 0.0,\n'
        '  "fp_tags": ["non_user_controlled", "safe_sink", "unreachable_prod"],\n'
        '  "evidence": [\n'
        '    {"path": "...", "line": "start-end", "snippet": "...", "why_it_matters": "..."}\n'
        "  ]\n"
        "}\n"
        f"draft: {json.dumps(state.get('draft', {}), ensure_ascii=True)}\n"
        f"answers: {json.dumps(state.get('answers', []), ensure_ascii=True)}\n"
        "If any question is unknown or evidence is weak, choose keep."
    )


def ensure_evidence(answers: List[Answer]) -> List[Answer]:
    cleaned: List[Answer] = []
    for ans in answers:
        evidence = ans.get("evidence") or []
        has_lines = any(e.get("line") for e in evidence if isinstance(e, dict))
        if not has_lines:
            cleaned.append({"question": ans.get("question", ""), "answer": "unknown", "evidence": []})
        else:
            cleaned.append(ans)
    return cleaned


class OverloadedError(RuntimeError):
    pass


def is_overloaded_error(exc: BaseException) -> bool:
    try:
        from google.genai.errors import ServerError  # type: ignore
        if isinstance(exc, ServerError):
            status_code = getattr(exc, "status_code", None)
            if status_code == 503:
                return True
    except Exception:
        pass
    status_code = getattr(exc, "status_code", None)
    if status_code == 503:
        return True
    text = str(exc)
    return "503" in text and ("overloaded" in text.lower() or "UNAVAILABLE" in text)


def build_graph(llm: Dict[str, LLMClient], rag: Any, config: Dict[str, Any]) -> Any:
    if StateGraph is None or END is None:
        raise RuntimeError("langgraph is required: pip install langgraph")

    def draft_fp_filter(state: CaseState) -> Dict[str, Any]:
        prompt = build_draft_prompt(state)
        response = llm["general"].complete(prompt)
        state["logger"].log("draft_fp_filter", prompt, response)
        data = extract_json(response)
        decision = data.get("decision") or "keep"
        reason = data.get("reason") or "insufficient evidence"
        return {"draft": {"decision": decision, "reason": reason}}

    def cove_generate_questions(state: CaseState) -> Dict[str, Any]:
        prompt = build_questions_prompt(state)
        response = llm["cove_questions"].complete(prompt)
        state["logger"].log("cove_generate_questions", prompt, response)
        data = extract_json(response)
        questions = data.get("questions") or []
        return {"questions": questions}

    def rag_query_planner(state: CaseState) -> Dict[str, Any]:
        prompt = build_query_plan_prompt(state)
        response = llm["general"].complete(prompt)
        state["logger"].log("rag_query_planner", prompt, response)
        data = extract_json(response)
        queries = data.get("queries") or []
        if not queries:
            symbols = tokenize(state.get("code_snippet", ""))
            hint = state.get("file_path") or None
            queries = [{"query": " ".join(symbols[:8]), "path_hint": hint}]
        return {"queries": queries}

    def repo_retrieve(state: CaseState) -> Dict[str, Any]:
        top_k = config["top_k"]
        alpha = config["alpha"]
        seen = set()
        results: List[Dict[str, Any]] = []
        for q in state.get("queries", []):
            query = q.get("query") or ""
            path_hint = q.get("path_hint")
            hits = rag.search(query, top_k=top_k, alpha=alpha, path_hint=path_hint)
            for hit in hits:
                key = (hit.path, hit.start_line, hit.end_line)
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "path": hit.path,
                    "start_line": hit.start_line,
                    "end_line": hit.end_line,
                    "snippet": hit.snippet,
                })
        return {"retrieved_context": results}

    def cove_answer_questions(state: CaseState) -> Dict[str, Any]:
        answers: List[Answer] = []
        context = state.get("retrieved_context", [])
        for question in state.get("questions", []):
            cache_key = build_cache_key(question, context)
            cached = state["cache"].get(cache_key)
            if cached is not None:
                answers.append(cached)
                continue
            prompt = build_answer_prompt(question, context)
            response = llm["cove_answers"].complete(prompt)
            state["logger"].log("cove_answer_questions", prompt, response)
            data = extract_json(response)
            raw_answers = data.get("answers") or []
            if raw_answers:
                ans = raw_answers[0]
            else:
                ans = {"question": question.get("question", ""), "answer": "unknown", "evidence": []}
            ans = ensure_evidence([ans])[0]
            state["cache"].set(cache_key, ans)
            answers.append(ans)
        return {"answers": answers}

    def refine_decision(state: CaseState) -> Dict[str, Any]:
        prompt = build_refine_prompt(state)
        response = llm["general"].complete(prompt)
        state["logger"].log("refine_decision", prompt, response)
        data = extract_json(response)
        decision = data.get("decision") or "keep"
        final = {
            "decision": decision,
            "drop_confidence": float(data.get("drop_confidence") or 0.0),
            "fp_tags": data.get("fp_tags") or [],
            "evidence": data.get("evidence") or [],
        }
        return {"final": final}

    def judge_gate(state: CaseState) -> Dict[str, Any]:
        final = state.get("final", {})
        if final.get("decision") != "drop_fp":
            return {"final": final}
        evidence = final.get("evidence") or []
        tags = set(final.get("fp_tags") or [])
        strong_tags = {"non_user_controlled", "safe_sink", "unreachable_prod"}
        confidence = float(final.get("drop_confidence") or 0.0)
        # Relaxed gate: allow drop with a single evidence or strong tag, as long as confidence meets threshold.
        if (len(evidence) < 1 and not (tags & strong_tags)) or confidence < config["confidence_threshold"]:
            final = {
                "decision": "keep",
                "drop_confidence": confidence,
                "fp_tags": list(tags),
                "evidence": evidence,
            }
        return {"final": final}

    graph = StateGraph(CaseState)
    graph.add_node("draft_fp_filter", draft_fp_filter)
    graph.add_node("cove_generate_questions", cove_generate_questions)
    graph.add_node("rag_query_planner", rag_query_planner)
    graph.add_node("repo_retrieve", repo_retrieve)
    graph.add_node("cove_answer_questions", cove_answer_questions)
    graph.add_node("refine_decision", refine_decision)
    graph.add_node("judge_gate", judge_gate)

    def route_after_draft(state: CaseState) -> str:
        if state.get("draft", {}).get("decision") == "drop_candidate":
            return "cove_generate_questions"
        return "refine_decision"

    graph.set_entry_point("draft_fp_filter")
    graph.add_conditional_edges("draft_fp_filter", route_after_draft)
    graph.add_edge("cove_generate_questions", "rag_query_planner")
    graph.add_edge("rag_query_planner", "repo_retrieve")
    graph.add_edge("repo_retrieve", "cove_answer_questions")
    graph.add_edge("cove_answer_questions", "refine_decision")
    graph.add_edge("refine_decision", "judge_gate")
    graph.add_edge("judge_gate", END)
    return graph.compile()


def build_cache_key(question: Question, context: List[Dict[str, Any]]) -> str:
    payload = json.dumps({"question": question, "context": context}, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_case(state: CaseState) -> Dict[str, Any]:
    final = state.get("final", {})
    return {
        "case_id": state.get("case_id"),
        "rule_id": state.get("rule_id"),
        "file": state.get("file_path"),
        "draft": state.get("draft"),
        "questions": state.get("questions"),
        "answers": state.get("answers"),
        "final": final,
    }


def load_issues(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "findings" in data:
        findings = data.get("findings")
        if not isinstance(findings, list):
            raise ValueError("issues JSON must contain a findings array")
        normalized = []
        for idx, finding in enumerate(findings, start=1):
            if not isinstance(finding, dict):
                continue
            if "issue_id" not in finding:
                finding = {**finding, "issue_id": f"finding_{idx}"}
            normalized.append(finding)
        return normalized
    if not isinstance(data, list):
        raise ValueError("issues JSON must be an array or contain findings")
    return data


def main() -> int:
    load_env_file(Path(__file__).resolve().parent / ".env")
    args = parse_args()
    env_model = os.environ.get("LLM_MODEL")
    if env_model:
        args.model = env_model
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = gemini_key
    repo_root = Path(args.repo).resolve()
    out_dir = Path(args.out_dir).resolve()
    index_dir = Path(args.index_dir).resolve()

    if not repo_root.exists():
        raise SystemExit(f"repo not found: {repo_root}")

    sys_path_added = False
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        sys_path_added = True

    try:
        from repo_rag import RepoRAG  # type: ignore
    except ImportError as exc:
        if sys_path_added:
            sys.path.pop(0)
        raise SystemExit(f"repo_rag import failed: {exc}")

    rag = RepoRAG.load(str(index_dir))
    if args.mock:
        llm: Dict[str, LLMClient] = {
            "general": MockLLMClient(),
            "cove_questions": MockLLMClient(),
            "cove_answers": MockLLMClient(),
        }
    else:
        llm = {
            "general": LangChainGeminiClient(
                model=args.model,
                temperature=args.temperature,
                system_prompt="You are a conservative SAST verifier.",
            ),
            "cove_questions": LangChainGeminiClient(
                model=args.model,
                temperature=args.temperature,
                system_prompt="You generate verification questions only.",
            ),
            "cove_answers": LangChainGeminiClient(
                model=args.model,
                temperature=args.temperature,
                system_prompt="You answer using retrieved repo context only.",
            ),
        }

    config = {
        "top_k": args.top_k,
        "alpha": args.alpha,
        "confidence_threshold": args.confidence_threshold,
    }
    graph = build_graph(llm, rag, config)

    issues = load_issues(Path(args.issues))
    if args.case_id:
        issues = [i for i in issues if str(i.get("issue_id")) == args.case_id]
        if not issues:
            raise SystemExit(f"case_id not found: {args.case_id}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache = QACache(out_dir / "cache" / "qa_cache.json")

    dropped: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    dropped_lock = threading.Lock()
    kept_lock = threading.Lock()
    skipped_lock = threading.Lock()

    def worker(case: Dict[str, Any]) -> None:
        case_input = build_case_input(repo_root, case)
        case_id = case_input["case_id"]
        logger = CaseLogger(out_dir / "logs" / f"{case_id}.jsonl")
        state: CaseState = {
            **case_input,
            "draft": {"decision": "keep", "reason": ""},
            "questions": [],
            "queries": [],
            "retrieved_context": [],
            "answers": [],
            "final": {"decision": "keep", "drop_confidence": 0.0, "fp_tags": [], "evidence": []},
            "logger": logger,
            "cache": cache,
        }
        try:
            final_state = graph.invoke(state)
        except OverloadedError as exc:
            print(f"[warn] skipped case {case_id}: {exc}", file=sys.stderr)
            with skipped_lock:
                skipped.append({
                    "case_id": case_id,
                    "rule_id": case_input.get("rule_id"),
                    "file": case_input.get("file_path"),
                    "reason": "model_overloaded",
                })
            return
        summary = summarize_case(final_state)
        decision = summary.get("final", {}).get("decision")
        if decision == "drop_fp":
            with dropped_lock:
                dropped.append(summary)
        else:
            with kept_lock:
                kept.append(summary)
        session_path = out_dir / "sessions" / f"{case_id}.json"
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    threads = max(1, int(args.threads))
    if threads == 1:
        for issue in issues:
            worker(issue)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=threads) as executor:
            list(executor.map(worker, issues))

    drop_path = out_dir / "llm_verified_drop_fp.json"
    keep_path = out_dir / "llm_verified_keep.json"
    skip_path = out_dir / "llm_verified_skipped.json"
    drop_path.write_text(json.dumps(dropped, ensure_ascii=True, indent=2), encoding="utf-8")
    keep_path.write_text(json.dumps(kept, ensure_ascii=True, indent=2), encoding="utf-8")
    skip_path.write_text(json.dumps(skipped, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"[done] drop_fp: {drop_path}")
    print(f"[done] keep: {keep_path}")
    print(f"[done] skipped: {skip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
