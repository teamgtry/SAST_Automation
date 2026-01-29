#!/usr/bin/env python3
# make_exclude_rules.py
#
# Reads semgrep dryrun JSON (from: semgrep scan --config auto --dryrun --json ...)
# and generates exclude_rules.txt (rule IDs) based on exclude_policy.yml.
#
# Your dryrun schema:
#   - rule id: results[*].check_id
#   - metadata: results[*].extra.metadata

import json
import sys
from pathlib import Path
import yaml


def load_policy(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def get_md(result: dict) -> dict:
    extra = result.get("extra") or {}
    md = extra.get("metadata") or {}
    return md if isinstance(md, dict) else {}


def has_any_metadata_keys(md: dict, keys) -> bool:
    for k in keys:
        if k in md and md[k] not in (None, [], "", {}):
            return True
    return False


def md_list_any_in(md: dict, field: str, wanted) -> bool:
    vals = as_list(md.get(field))
    wanted_set = set(wanted)
    for v in vals:
        if str(v) in wanted_set:
            return True
    return False


def should_keep(md: dict, policy: dict) -> bool:
    keep = policy.get("keep", {}) or {}
    req_any = keep.get("require_any_metadata_keys", []) or []
    if req_any:
        return has_any_metadata_keys(md, req_any)
    return False


def is_audit(md: dict) -> bool:
    subs = as_list(md.get("subcategory"))
    return any(str(s) == "audit" for s in subs)


def confidence_is_low(md: dict) -> bool:
    c = md.get("confidence")
    return isinstance(c, str) and c.upper() == "LOW"


def likelihood_is_low(md: dict) -> bool:
    v = md.get("likelihood")
    return isinstance(v, str) and v.upper() == "LOW"


def impact_is_low(md: dict) -> bool:
    v = md.get("impact")
    return isinstance(v, str) and v.upper() == "LOW"


def should_exclude(md: dict, policy: dict) -> bool:
    # 1) audit 룰인데 metadata가 비어있으면 제외하지 않음
    if is_audit(md) and not md:
        return False
    
    ex = policy.get("exclude", {}) or {}

    # 2) audit + confidence LOW
    if ex.get("audit_low_only", False):
        if is_audit(md) and confidence_is_low(md):
            return True

    # 2-1) audit + (likelihood LOW OR impact LOW)
    if ex.get("audit_low_likelihood_or_impact", False):
        if is_audit(md) and (likelihood_is_low(md) or impact_is_low(md)):
            return True

    # 3) crypto (vulnerability_class is a list in your output)
    vc_any = ex.get("vulnerability_class_any_in", []) or []
    if vc_any and md_list_any_in(md, "vulnerability_class", vc_any):
        return True

    # 4) optional categories (some rules may have category != security)
    cat_opt = ex.get("category_in_optional", []) or []
    if cat_opt:
        cat = md.get("category")
        if isinstance(cat, str) and cat in set(cat_opt):
            return True

    return False


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun-json", default="dryrun_auto.json")
    ap.add_argument("--policy", default="exclude_policy.yml")
    ap.add_argument("--out", default="exclude_rules.txt")
    args = ap.parse_args()

    dry = json.loads(Path(args.dryrun_json).read_text(encoding="utf-8"))
    policy = load_policy(Path(args.policy))

    results = dry.get("results")
    if not isinstance(results, list):
        print("[-] dryrun json has no 'results' list", file=sys.stderr)
        sys.exit(2)

    exclude_overrides_keep = bool((policy.get("decision", {}) or {}).get("exclude_overrides_keep", True))

    total_ids = set()
    exclude_ids = set()

    for res in results:
        if not isinstance(res, dict):
            continue

        rid = res.get("check_id")
        if not rid:
            continue

        total_ids.add(rid)
        md = get_md(res)

        keep = should_keep(md, policy)
        exclude = should_exclude(md, policy)

        if exclude_overrides_keep:
            if exclude:
                exclude_ids.add(rid)
        else:
            if exclude and not keep:
                exclude_ids.add(rid)

    total = len(total_ids)
    exc = len(exclude_ids)
    ratio = (exc / total) if total else 0.0

    max_ratio = float((policy.get("safety", {}) or {}).get("max_exclude_ratio", 1.0))
    if total and ratio > max_ratio:
        print(f"[-] Safety trip: exclude ratio {ratio:.2%} > {max_ratio:.2%}", file=sys.stderr)
        print(f"    total unique rule ids={total}, excluded={exc}", file=sys.stderr)
        sys.exit(3)

    Path(args.out).write_text("\n".join(sorted(exclude_ids)) + ("\n" if exclude_ids else ""), encoding="utf-8")

if __name__ == "__main__":
    main()
