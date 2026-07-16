"""
backend/models.py
─────────────────
Pydantic request/response models for the CFR Poker Solver REST API.
"""

from pydantic import BaseModel


class TrainRequest(BaseModel):
    game: str
    iterations: int | None = None


class TrainResponse(BaseModel):
    game: str
    iterations: int
    nodes: int
    final_ev: float


class StrategyResponse(BaseModel):
    game: str
    strategy: dict[str, dict[str, float]]


class InfosetResponse(BaseModel):
    game: str
    infoset: str
    strategy: dict[str, float]


class ExploitabilityResponse(BaseModel):
    game: str
    exploitability: float


class EVResponse(BaseModel):
    game: str
    ev: float


class HealthResponse(BaseModel):
    status: str
    trained_games: list[str]
