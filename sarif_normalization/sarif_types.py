from typing import Optional, List, TypedDict, Union


class Region(TypedDict, total=False):
    Line: Optional[Union[int, str]]


class PrimaryLocation(TypedDict, total=False):
    uri: Optional[str]
    region: Region


class RelatedLocation(TypedDict, total=False):
    uri: Optional[str]
    region: Region


class Locations(TypedDict, total=False):
    primary: PrimaryLocation
    related: Optional[List[RelatedLocation]]


class Tool(TypedDict, total=False):
    name: Optional[str]


class Rule(TypedDict, total=False):
    id: str
    description: Optional[str]
    tags: Optional[List[str]]


class NormalizedIssue(TypedDict, total=False):
    issue_id: str
    tool: Tool
    rule: Rule
    message: Optional[str]
    locations: Locations