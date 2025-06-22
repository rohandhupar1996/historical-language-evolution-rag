# ==========================================
# FILE: access_pipeline/models.py
# ==========================================
"""Pydantic models for API."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class QueryRequest(BaseModel):
    query: str
    period: Optional[str] = None
    genre: Optional[str] = None
    limit: Optional[int] = 100


class EvolutionQuery(BaseModel):
    word: str
    start_period: str
    end_period: str


class SearchRequest(BaseModel):
    query: str
    period: Optional[str] = None
    limit: Optional[int] = 50


class EvolutionResponse(BaseModel):
    word: str
    period_range: str
    evolution: List[Dict[str, Any]]


class LinguisticAnalysisResponse(BaseModel):
    filters: Dict[str, Optional[str]]
    results: List[Dict[str, Any]]


class SearchResponse(BaseModel):
    query: str
    period_filter: Optional[str]
    results: List[Dict[str, Any]]
