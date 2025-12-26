from typing import Any, List, Dict


def extract_primary_location(result: Any) -> Dict[str, Any]:
    """
    Extract primary location from result.locations[0]
    Implements endLine fallback: endLine or startLine
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
    
    return {
        "uri": uri,
        "region": {
            "startLine": start_line,
            "endLine": end_line,
        },
    }


def extract_related_locations(result: Any) -> List[Dict[str, Any]]:
    """
    Flatten code flows into related locations
    Implements: codeFlows[*].threadFlows[*].locations[*].location.physicalLocation
    """
    related: List[Dict[str, Any]] = []
    
    if not isinstance(result, dict):
        return related
    
    code_flows = result.get("codeFlows", [])
    
    if not isinstance(code_flows, list):
        return related
    
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
                
                related.append({
                    "uri": artifact_location.get("uri"),
                    "region": {
                        "startLine": region.get("startLine"),
                    },
                    "role": "data_flow",
                })
    
    return related