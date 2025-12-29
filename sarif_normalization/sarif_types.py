from typing import Optional, List, TypedDict, Literal


class Region(TypedDict, total=False):
    startLine: Optional[int]
    endLine: Optional[int]


class PrimaryLocation(TypedDict, total=False):
    uri: Optional[str]
    region: Region


class RelatedLocation(TypedDict, total=False):
    uri: Optional[str]
    region: Region
    role: Literal["data_flow"]


class Locations(TypedDict, total=False):
    primary: PrimaryLocation
    related: Optional[List[RelatedLocation]]


class Tool(TypedDict, total=False):
    name: Optional[str]
    version: Optional[str]
    run_index: int


class Rule(TypedDict, total=False):
    id: str
    name: Optional[str]
    description: Optional[str]
    tags: Optional[List[str]]


class Sast(TypedDict, total=False):
    severity: Optional[str]


class NormalizedIssue(TypedDict, total=False):
    issue_id: str
    tool: Tool
    rule: Rule
    sast: Sast
    message: Optional[str]
    locations: Locations