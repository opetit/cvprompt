from typing import List
from pydantic import BaseModel


class ScoredChunk(BaseModel):
    project_id: int
    score: float
    content: str


class ScoredProject(BaseModel):
    id: int
    score: float
    name: str
    company: str
    description: str


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    chunks: List[ScoredChunk]
    projects: List[ScoredProject]
