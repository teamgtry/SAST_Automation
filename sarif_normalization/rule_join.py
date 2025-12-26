from typing import Any, Optional


def find_rule_by_id(run: Any, rule_id: str) -> Optional[dict]:
    if not isinstance(run, dict):
        return None
    
    tool = run.get("tool", {})
    driver = tool.get("driver", {})
    rules = driver.get("rules", [])
    
    if not isinstance(rules, list):
        return None
    
    for rule in rules:
        if isinstance(rule, dict) and rule.get("id") == rule_id:
            return rule
    
    return None