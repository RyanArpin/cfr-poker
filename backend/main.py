"""
backend/main.py
───────────────
FastAPI REST API for the CFR Poker Solver.

Serves trained GTO strategies for Kuhn Poker and Leduc Poker.
Endpoints allow training, querying strategies per infoset, and
computing exploitability.

Run from project root:
    PYTHONPATH=. python backend/run.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from kuhn.kuhn_poker import KuhnPoker
from kuhn.cfr import CFRTrainer
from kuhn.analysis import exploitability as kuhn_exploitability

from leduc.leduc_poker import LeducPoker
from leduc.cfr import LeducCFRTrainer
from leduc.analysis import exploitability as leduc_exploitability

from backend.models import (
    TrainRequest, TrainResponse,
    StrategyResponse, InfosetResponse,
    ExploitabilityResponse, EVResponse, HealthResponse,
)


# ─────────────────────────────────────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CFR Poker Solver API",
    description="REST API for querying GTO strategies computed by Counterfactual "
                "Regret Minimization on Kuhn Poker and Leduc Poker.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level state — stores trained models
# ─────────────────────────────────────────────────────────────────────────────

trained_trainers: dict[str, object] = {}
trained_strategies: dict[str, dict] = {}
trained_ev: dict[str, float] = {}

# Iteration caps to prevent accidental long-running requests
ITER_DEFAULTS = {"kuhn": 10_000, "leduc": 1_000}
ITER_CAPS     = {"kuhn": 100_000, "leduc": 10_000}


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest):
    """
    Train a CFR solver for the specified game.

    Body: { "game": "kuhn" | "leduc", "iterations": 10000 }
    """
    game = request.game.lower()
    if game not in ("kuhn", "leduc"):
        raise HTTPException(400, f"Unknown game '{game}'. Use 'kuhn' or 'leduc'.")

    iterations = request.iterations or ITER_DEFAULTS[game]
    iterations = min(iterations, ITER_CAPS[game])

    if game == "kuhn":
        trainer = CFRTrainer()
    else:
        trainer = LeducCFRTrainer()

    ev_history = trainer.train(iterations=iterations)
    strategy   = trainer.get_strategy()

    trained_trainers[game]   = trainer
    trained_strategies[game] = strategy
    trained_ev[game]         = ev_history[-1]

    return TrainResponse(
        game=game,
        iterations=iterations,
        nodes=len(trainer.nodes),
        final_ev=round(ev_history[-1], 6),
    )


@app.get("/strategy/{game}", response_model=StrategyResponse)
def get_strategy(game: str):
    """Return the full trained strategy for a game."""
    game = game.lower()
    if game not in trained_strategies:
        raise HTTPException(404, f"Game '{game}' not yet trained. POST /train first.")

    return StrategyResponse(game=game, strategy=trained_strategies[game])


@app.get("/strategy/{game}/{infoset}", response_model=InfosetResponse)
def get_infoset_strategy(game: str, infoset: str):
    """Return the strategy for a single infoset (URL-encoded)."""
    game = game.lower()
    if game not in trained_strategies:
        raise HTTPException(404, f"Game '{game}' not yet trained. POST /train first.")

    strategy = trained_strategies[game]
    if infoset not in strategy:
        raise HTTPException(404, f"Infoset '{infoset}' not found in {game} strategy.")

    return InfosetResponse(game=game, infoset=infoset, strategy=strategy[infoset])


@app.get("/exploitability/{game}", response_model=ExploitabilityResponse)
def get_exploitability(game: str):
    """Compute exploitability of the trained strategy."""
    game = game.lower()
    if game not in trained_strategies:
        raise HTTPException(404, f"Game '{game}' not yet trained. POST /train first.")

    strategy = trained_strategies[game]
    if game == "kuhn":
        exploit = kuhn_exploitability(KuhnPoker(), strategy)
    else:
        exploit = leduc_exploitability(LeducPoker(), strategy)

    return ExploitabilityResponse(game=game, exploitability=round(exploit, 6))


@app.get("/ev/{game}", response_model=EVResponse)
def get_ev(game: str):
    """Return the final EV from training."""
    game = game.lower()
    if game not in trained_ev:
        raise HTTPException(404, f"Game '{game}' not yet trained. POST /train first.")

    return EVResponse(game=game, ev=round(trained_ev[game], 6))


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — reports which games are currently trained."""
    return HealthResponse(
        status="ok",
        trained_games=list(trained_strategies.keys()),
    )
