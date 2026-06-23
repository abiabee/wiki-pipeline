"""Pydantic models for leaf JSONs.

The schema is intentionally permissive (extra fields allowed, optionals where
the data shows real-world variation) so that validation surfaces *real*
problems and not arbitrary stylistic noise. Hard rules live in
`validate_leaves.py`; here we just describe the shape.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


_BASE_CONFIG = ConfigDict(extra="allow", populate_by_name=True)


class LeafSource(BaseModel):
    model_config = _BASE_CONFIG

    drive_file_id: str
    name: str
    url: str = ""
    mime_type: str = "unknown"
    last_modified: Optional[str] = None
    last_ingested: Optional[str] = None


class LeafClassification(BaseModel):
    model_config = _BASE_CONFIG

    document_type: str = "unknown"
    business_area: List[str] = Field(default_factory=list)
    audience: List[str] = Field(default_factory=list)
    sensitivity: str = "unknown"
    status: str = "unknown"


class LeafSummary(BaseModel):
    model_config = _BASE_CONFIG

    one_sentence: str = ""
    short_summary: str = ""
    key_points: List[str] = Field(default_factory=list)


class LeafFact(BaseModel):
    model_config = _BASE_CONFIG

    claim: str
    confidence: Optional[str] = None
    evidence: Optional[str] = None
    location: Optional[str] = None


class LeafEntities(BaseModel):
    model_config = _BASE_CONFIG

    customers: List[str] = Field(default_factory=list)
    competitors: List[str] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    erps: List[str] = Field(default_factory=list)
    partners: List[str] = Field(default_factory=list)
    people: List[str] = Field(default_factory=list)
    policies: List[str] = Field(default_factory=list)


class LeafRelationship(BaseModel):
    model_config = _BASE_CONFIG

    # `from` is reserved in Python; map it via alias.
    from_: str = Field(alias="from")
    relationship: str
    to: str
    evidence: Optional[str] = None


class LeafEmbedding(BaseModel):
    model_config = _BASE_CONFIG

    title: str
    text: str
    keywords: List[str] = Field(default_factory=list)


class LeafPromotion(BaseModel):
    model_config = _BASE_CONFIG

    ready_for_clustering: bool
    notes: str = ""


class Leaf(BaseModel):
    model_config = _BASE_CONFIG

    leaf_id: str
    source: LeafSource
    classification: LeafClassification
    summary: LeafSummary
    facts: List[LeafFact] = Field(default_factory=list)
    entities: LeafEntities
    relationships: List[LeafRelationship] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    contradictions: List[Any] = Field(default_factory=list)
    embedding: LeafEmbedding
    promotion: LeafPromotion
