import { Codex } from "@openai/codex-sdk";
import fs from "node:fs";
import path from "node:path";

type Verdict = "true_positive" | "false_positive" | "needs_review";

const properties = {
  issue_id: { type: "string" },
  verdict: { type: "string", enum: ["true_positive", "false_positive", "needs_review"] },
  confidence: { type: "number", minimum: 0, maximum: 1 },
  summary: { type: "string" },
  reasoning: { type: "string" },
  evidence: {
    type: "array",
    items: {
      type: "object",
      properties: {
        path: { type: "string" },
        lines: { type: "string" },
        snippet: { type: "string" },
        why_it_matters: { type: "string" },
      },
      required: ["path", "lines", "snippet", "why_it_matters"],
      additionalProperties: false,
    },
  },
  recommended_fix: { type: "string" },
  poc: { type: "string" },

  //optional로 두면 스키마 검증에서 걸리니까, 그냥 항상 넣게 만들자(빈 문자열 가능)
  notes: { type: "string" },
} as const;

const outputSchema = {
  type: "object",
  properties,
  required: Object.keys(properties), /
  additionalProperties: false,
} as const;

function parseArgs() {
  const args = process.argv.slice(2);
  const m = new Map<string, string>();

  for (let i = 0; i < args.length; i++) {
    const key = args[i];
    if (!key.startsWith("--")) continue;

    const next = args[i + 1];
    if (!next || next.startsWith("--")) {
      // flag 형태: --skip-git-check 처럼 값 없이 들어오면 "true" 취급
      m.set(key, "true");
    } else {
      m.set(key, next);
      i++;
    }
  }

  const repo = m.get("--repo");
  const issues = m.get("--issues");
  const outDir = m.get("--out-dir");
  const threadsStr = m.get("--threads") ?? "1";
  const issueId = m.get("--issue-id") ?? undefined;
  const skipGitCheck = (m.get("--skip-git-check") ?? "false").toLowerCase() === "true";

  if (!repo || !issues || !outDir) {
    throw new Error(
      "usage: --repo <dir> --issues <normalized.json> --out-dir <dir> [--threads N] [--issue-id ID] [--skip-git-check true]"
    );
  }

  const threads = Math.max(1, Number.parseInt(threadsStr, 10) || 1);

  return { repo, issues, outDir, threads, issueId, skipGitCheck };
}

function buildPrompt(issue: any) {
  return `
You are a security triage agent.

Output requirements:
- Always include "notes" (use "" if no extra notes).
- Always include "poc" (if verdict is true_positive, provide minimal exploit steps/request; otherwise use "").

Goal:
- Determine whether the finding is a TRUE POSITIVE (real security issue),
  FALSE POSITIVE (not exploitable / not a bug), or NEEDS_REVIEW (insufficient evidence).
- You MUST inspect the repository code directly (open files, follow calls/usages as needed).
- If the report points to spec/test files, confirm whether the same pattern exists in production paths.

Output:
- Return ONLY the final JSON that conforms to the provided JSON Schema.

Normalized finding JSON:
${JSON.stringify(issue, null, 2)}

Checklist:
1) Open the primary location file and inspect the referenced line(s).
2) Track data flow: user-controlled source -> sink.
3) Follow at least one hop of the call chain: check the caller and callee files for this code path to see if mitigation exists there.
4) Identify mitigations: validation, allowlist, escaping, parameterization, authn/authz, etc.
5) Provide evidence snippets: path + line range + code + why it matters.
6) Give a minimal recommended_fix.
7) For true_positive, add a minimal PoC (commands/HTTP request/steps) that demonstrates exploitability in the "poc" field; otherwise set "poc" to "".
`.trim();
}

async function readJsonFile(p: string) {
  const txt = await fs.promises.readFile(p, "utf-8");
  return JSON.parse(txt);
}

async function ensureDir(p: string) {
  await fs.promises.mkdir(p, { recursive: true });
}

async function triageOne(codex: Codex, repoDirAbs: string, issue: any, skipGitCheck: boolean) {
  const thread = codex.startThread({
    workingDirectory: repoDirAbs,
    ...(skipGitCheck ? { skipGitRepoCheck: true } : {}),
  });

  const prompt = buildPrompt(issue);

  const turn = await thread.run(prompt, { outputSchema });
  const final = turn.finalResponse;

  const obj = typeof final === "string" ? JSON.parse(final) : final;

  // 최소 sanity 체크
  if (!obj || obj.issue_id == null || obj.verdict == null) {
    throw new Error("Codex output did not match expected structure");
  }
  return obj as {
    issue_id: string;
    verdict: Verdict;
    confidence: number;
    summary: string;
    reasoning: string;
    evidence: Array<{ path: string; lines: string; snippet: string; why_it_matters: string }>;
    recommended_fix: string;
    poc: string;
    notes?: string;
  };
}

async function main() {
  const { repo, issues, outDir, threads, issueId, skipGitCheck } = parseArgs();

  const repoDirAbs = path.resolve(repo);
  const outDirAbs = path.resolve(outDir);

  await ensureDir(outDirAbs);

  const allIssues = await readJsonFile(path.resolve(issues));
  if (!Array.isArray(allIssues)) throw new Error("issues JSON must be an array");

  const targetIssues = issueId
    ? allIssues.filter((x: any) => x?.issue_id === issueId)
    : allIssues;

  if (issueId && targetIssues.length === 0) {
    throw new Error(`issue_id not found: ${issueId}`);
  }

  const codex = new Codex();

  let cursor = 0;
  const resultsIndex: Array<{
    issue_id: string;
    status: "ok" | "error";
    verdict?: Verdict;
    confidence?: number;
    summary?: string;
    reasoning?: string;
    evidence?: Array<{ path: string; lines: string; snippet: string; why_it_matters: string }>;
    recommended_fix?: string;
    poc?: string;
    notes?: string;
    error?: string;
  }> = [];

  // 워커(=동시 thread) 함수
  async function worker(workerId: number) {
    while (true) {
      const i = cursor;
      cursor++;
      if (i >= targetIssues.length) return;

      const issue = targetIssues[i];
      const id = String(issue?.issue_id ?? `index-${i}`);

      try {
        // 이슈 1개 = thread 1개
        const verdictObj = await triageOne(codex, repoDirAbs, issue, skipGitCheck);

        resultsIndex.push({
          issue_id: id,
          status: "ok",
          verdict: verdictObj.verdict,
          confidence: verdictObj.confidence,
          summary: verdictObj.summary,
          reasoning: verdictObj.reasoning,
          evidence: verdictObj.evidence,
          recommended_fix: verdictObj.recommended_fix,
          poc: verdictObj.poc ?? "",
          notes: verdictObj.notes ?? "",
        });

        console.log(`[worker ${workerId}] ok: ${id} -> ${verdictObj.verdict} (${verdictObj.confidence})`);
      } catch (e: any) {
        resultsIndex.push({
          issue_id: id,
          status: "error",
          error: e?.message ?? String(e),
        });
        console.error(`[worker ${workerId}] error: ${id} -> ${e?.message ?? e}`);
      }
    }
  }

  const workerCount = Math.min(threads, targetIssues.length || 1);
  console.log(`[info] issues=${targetIssues.length}, threads=${workerCount}, outDir=${outDirAbs}`);

  await Promise.all(Array.from({ length: workerCount }, (_, k) => worker(k)));

  const positivePath = path.join(outDirAbs, "llm_verified_result_positive.json");
  const falsePath = path.join(outDirAbs, "llm_verified_result_false.json");

  const truePositives = resultsIndex.filter(
    (r) => r.status === "ok" && r.verdict === "true_positive"
  );
  const falsePositives = resultsIndex.filter(
    (r) => r.status === "ok" && r.verdict === "false_positive"
  );

  await Promise.all([
    fs.promises.writeFile(positivePath, JSON.stringify(truePositives, null, 2), "utf-8"),
    fs.promises.writeFile(falsePath, JSON.stringify(falsePositives, null, 2), "utf-8"),
  ]);

  console.log(`[done] wrote positives: ${positivePath}`);
  console.log(`[done] wrote false positives: ${falsePath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
