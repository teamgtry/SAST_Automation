from typing import Any, List, Dict, Union


def extract_primary_location(result: Any) -> Dict[str, Any]:
    """
    Extract primary location from result.locations[0]
    Implements Line consolidation: if startLine == endLine, use single number,
    otherwise use "startLine-endLine" format
    """
    physical_location = {}
    
    if isinstance(result, dict):
        locations = result.get("locations", [])
        if isinstance(locations, list) and len(locations) > 0:
            physical_location = locations[0].get("physicalLocation", {})
    
    region = physical_location.get("region", {})
    start_line = region.get("startLine")
    end_line = region.get("endLine", start_line)
    
    artifact_location = physical_location.get("artifactLocation", {})
    uri = artifact_location.get("uri")
    
    # Consolidate Line field
    line_value: Union[int, str, None] = None
    if start_line is not None:
        if end_line is not None and start_line != end_line:
            line_value = f"{start_line}-{end_line}"
        else:
            line_value = start_line
    
    return {
        "uri": uri,
        "region": {
            "Line": line_value,
        },
    }


def extract_related_locations(result: Any) -> List[Dict[str, Any]]:
    """
    Flatten code flows into related locations
    Groups by uri and aggregates Line numbers
    Removes role field and deduplicates
    """
    if not isinstance(result, dict):
        return []
    
    code_flows = result.get("codeFlows", [])
    
    if not isinstance(code_flows, list):
        return []
    
    # Collect all locations first
    raw_locations: List[Dict[str, Any]] = []
    
    for code_flow in code_flows:
        if not isinstance(code_flow, dict):
            continue
        
        thread_flows = code_flow.get("threadFlows", [])
        
        if not isinstance(thread_flows, list):
            continue
        
        for thread_flow in thread_flows:
            if not isinstance(thread_flow, dict):
                continue
            
            locations = thread_flow.get("locations", [])
            
            if not isinstance(locations, list):
                continue
            
            for loc in locations:
                if not isinstance(loc, dict):
                    continue
                
                location = loc.get("location", {})
                physical_location = location.get("physicalLocation", {})
                
                if not physical_location:
                    continue
                
                artifact_location = physical_location.get("artifactLocation", {})
                region = physical_location.get("region", {})
                
                uri = artifact_location.get("uri")
                start_line = region.get("startLine")
                
                if uri and start_line is not None:
                    raw_locations.append({
                        "uri": uri,
                        "line": start_line,
                    })
    
    # Group by uri and aggregate lines
    uri_to_lines: Dict[str, List[int]] = {}
    for loc in raw_locations:
        uri = loc["uri"]
        line = loc["line"]
        
        if uri not in uri_to_lines:
            uri_to_lines[uri] = []
        
        # Deduplicate lines for same uri
        if line not in uri_to_lines[uri]:
            uri_to_lines[uri].append(line)
    
    # Build final related locations
    related: List[Dict[str, Any]] = []
    for uri, lines in uri_to_lines.items():
        # Sort lines for consistent output
        sorted_lines = sorted(lines)
        
        # Format as comma-separated string
        line_str = ", ".join(str(line) for line in sorted_lines)
        
        related.append({
            "uri": uri,
            "region": {
                "Line": line_str,
            },
        })
    
    return related