from typing import Any, List, Optional
import logging

# Import from local modules
from .types import NormalizedIssue
from .rule_join import find_rule_by_id
from .extractors import extract_primary_location, extract_related_locations


logger = logging.getLogger(__name__)


class SarifParser:    
    def parse(self, sarif: dict) -> List[NormalizedIssue]:
        if not isinstance(sarif, dict):
            return []
        
        runs = sarif.get("runs", [])
        
        if not isinstance(runs, list):
            return []
        
        issues: List[NormalizedIssue] = []
        
        for run_index, run in enumerate(runs):
            if not isinstance(run, dict):
                continue
            
            results = run.get("results", [])
            
            if not isinstance(results, list):
                continue
            
            for result_index, result in enumerate(results):
                try:
                    issue = self._parse_result(run, result, run_index, result_index)
                    if issue:
                        issues.append(issue)
                except Exception as error:
                    logger.error(
                        f"Failed to parse result {result_index} in run {run_index}: {error}"
                    )
        
        return issues
    
    def _parse_result(
        self,
        run: dict,
        result: dict,
        run_index: int,
        result_index: int,
    ) -> Optional[NormalizedIssue]:
        if not isinstance(result, dict):
            return None
        
        rule_id = result.get("ruleId")
        
        if not rule_id:
            logger.warning(f"Missing ruleId at run {run_index}, result {result_index}")
            return None
        
        # Find matching rule definition (implements "where" clause)
        rule = find_rule_by_id(run, rule_id)
        
        # Extract locations using utility functions
        primary_location = extract_primary_location(result)
        related_locations = extract_related_locations(result)
        
        # Sanitize rule_id for issue_id (replace / with -)
        sanitized_rule_id = rule_id.replace("/", "-")
        
        # Get tool information
        tool_info = run.get("tool", {})
        driver = tool_info.get("driver", {})
        tool_name = driver.get("name", "unknown").lower()
        
        # Get message
        message_obj = result.get("message", {})
        message_text = message_obj.get("text") if isinstance(message_obj, dict) else None
        
        # Build normalized issue according to schema
        issue: NormalizedIssue = {
            "issue_id": f"{tool_name}-run{run_index}-{sanitized_rule_id}-{result_index}",
            "tool": {
                "name": driver.get("name"),
                "version": driver.get("version"),
                "run_index": run_index,
            },
            "rule": {
                "id": rule_id,
                "name": rule.get("name") if rule else None,
                "description": (
                    rule.get("fullDescription", {}).get("text")
                    if rule and isinstance(rule.get("fullDescription"), dict)
                    else None
                ),
                "tags": (
                    rule.get("properties", {}).get("tags")
                    if rule and isinstance(rule.get("properties"), dict)
                    else None
                ),
            },
            "sast": {
                "severity": result.get("level"),
            },
            "message": message_text,
            "locations": {
                "primary": primary_location,
            },
        }
        
        # Optional: only include related if exists
        if related_locations:
            issue["locations"]["related"] = related_locations
        
        return issue


def parse_sarif(sarif: dict) -> List[NormalizedIssue]:
    parser = SarifParser()
    return parser.parse(sarif)