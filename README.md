# CFR Poker Solver

Counterfactual Regret Minimization (CFR) implemented from scratch in Python, following Zinkevich et al. (2007) "Regret Minimization in Games with Incomplete Information." The project builds through two poker variants — Kuhn Poker and Leduc Poker — then wraps the solver in a REST API for interactive queries.

**Student project:** First year Honours Applied Mathematics, University of Waterloo.

---

## Demo

### Kuhn Poker

![Kuhn Poker demo](docs/kuhn-demo.gif)

### Leduc Poker

![Leduc Poker demo](docs/leduc-demo.gif)

**Live demo:** [cfr-poker.vercel.app](https://cfr-poker.vercel.app) — *the backend runs on Render's free tier and sleeps after inactivity, so the first "Train" request can take ~50s to wake up.*

---

## Live Demo

- **Frontend:** [cfr-poker.vercel.app](https://cfr-poker.vercel.app) *(placeholder — update after deploy)*
- **API docs:** [cfr-poker-api.onrender.com/docs](https://cfr-poker-api.onrender.com/docs) *(placeholder — update after deploy)*

---

## What is CFR?

CFR is the algorithm behind modern poker solvers (PioSOLVER, GTO+, etc.). The idea: play a game thousands of times, track how much you *regret* not having taken a different action at each decision point, and shift your strategy toward actions you regret not taking. The average strategy over all iterations converges to a Nash equilibrium — proven in Theorem 3 of Zinkevich et al.

### Core notation (following the paper)

| Symbol | Meaning |
|---|---|
| `I` | Information set — all game states a player cannot distinguish |
| `σ(I, a)` | Strategy: probability of taking action `a` at infoset `I` |
| `π^σ_{-i}(I)` | Counterfactual reach: opponent's contribution to reaching `I` |
| `R^T_i(I, a)` | Cumulative counterfactual regret for action `a` at infoset `I` |
| `σ̄^T_i` | Average strategy over `T` iterations — this converges to Nash |

### Regret matching

```
σ(I, a) = R⁺(I, a) / Σ_b R⁺(I, b)
```

where `R⁺ = max(R, 0)`. If all regrets are non-positive, play uniformly.

### Two-pass design (correctness requirement)

All nodes must see the **same strategy profile** within one iteration. Updating regrets mid-sweep makes each iteration order-dependent. This implementation uses a strict two-pass design:

1. **Freeze** the current strategy profile as a snapshot
2. **Collect** all regret and strategy deltas across all deals — without writing to nodes
3. **Apply** all updates atomically after the full sweep

---

## Nash Equilibrium Reference

### Kuhn Poker — analytical equilibrium

Unlike most games, Kuhn Poker has a *known closed-form* Nash equilibrium (Kuhn, 1950). It is not a single strategy but a one-parameter family, parameterized by a bluff frequency `α ∈ [0, 1/3]`.

| Infoset | Meaning | Nash Strategy |
|---|---|---|
| `J:` | P1 opens with Jack | bet with prob `α` (≤ 1/3), else check |
| `Q:` | P1 opens with Queen | always check |
| `K:` | P1 opens with King | bet with prob `3α` |
| `J:b` | P1 Jack facing a bet | always fold |
| `Q:b` | P1 Queen facing a bet | call with prob 1/3 |
| `K:b` | P1 King facing a bet | always call |
| `J:c` | P1 Jack after opponent checks | bet with prob 1/3 |
| `Q:c` | P1 Queen after opponent checks | always check |
| `K:c` | P1 King after opponent checks | always bet |
| `Q:cb` | P1 Queen facing check-then-bet | call with prob 1/3 |

The solver converges to **one member** of this equilibrium family (which specific `α` depends on the run), and the game value is exactly **−1/18 ≈ −0.0556** for Player 1. Acting first is a slight disadvantage — a known counterintuitive result. The CFR-trained strategy in this repo matches this equilibrium to within an exploitability of **~0.005**.

### Leduc Poker — no closed form

Unlike Kuhn, Leduc Poker has **no closed-form Nash equilibrium**. The two-round structure with a community card dealt between rounds makes it analytically intractable — which is exactly why an iterative solver like CFR is valuable. This implementation discovers **~288 information sets** and converges to a low-exploitability approximate equilibrium. The learned strategy shows the expected qualitative behaviour: aggression spikes when the private card pairs with the community card (visible in the Community Card Effect plot below).

---

## Project Structure

```
cfr-poker/
├── kuhn/
│   ├── kuhn_poker.py     # Game environment: rules, payoffs, infoset keys
│   ├── cfr.py            # CFR algorithm: Node, CFRTrainer, Nash reference values
│   ├── analysis.py       # Exploitability, best response, convergence plots
│   └── plots/
│       ├── convergence.png
│       └── strategy_heatmap.png
├── leduc/
│   ├── leduc_poker.py    # Game environment: 2-round, community card, pair logic
│   ├── cfr.py            # LeducCFRTrainer with chance-node enumeration
│   ├── analysis.py       # Exploitability, strategy heatmaps, community card effect
│   └── plots/
│       ├── convergence.png
│       ├── strategy_by_round.png
│       └── community_card_effect.png
├── backend/
│   ├── main.py           # FastAPI app — REST endpoints for querying GTO strategies
│   ├── models.py         # Pydantic request/response models
│   └── run.py            # Uvicorn entry point
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Root component — holds all state, wires everything together
│   │   ├── api.ts            # Typed fetch wrappers for the backend REST API
│   │   ├── types.ts          # Shared TypeScript types (Game, StrategyResponse, etc.)
│   │   └── components/
│   │       ├── GameSelector.tsx     # Kuhn / Leduc toggle
│   │       ├── TrainButton.tsx      # Triggers training, shows spinner
│   │       ├── CardSelector.tsx     # Private + community card picker
│   │       ├── HistoryBuilder.tsx   # Legal-action history builder with breadcrumb
│   │       ├── StrategyDisplay.tsx  # GTO probability bars per action
│   │       └── StatsBar.tsx         # Nodes / EV / exploitability
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── vercel.json           # SPA routing config for Vercel
├── render.yaml           # Render deployment config for the backend
├── tests/
│   ├── test_kuhn_poker.py    # 34 tests — game environment
│   ├── test_cfr_trainer.py   # 18 tests — CFR algorithm and Nash convergence
│   └── test_leduc.py         # 72 tests — Leduc environment, trainer, analysis
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/RyanArpin/cfr-poker.git
cd cfr-poker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Always run scripts from the project root with `PYTHONPATH=.` to avoid import errors.

---

## Phase 1 — Kuhn Poker Environment

Kuhn Poker is the simplest non-trivial poker game, introduced by Harold Kuhn (1950) and the standard CFR benchmark.

**Rules:**
- Deck: Jack (J=0), Queen (Q=1), King (K=2). Each player gets one card; one is burned.
- Both ante 1 chip. One betting round: check (`c`) or bet (`b`), 1 chip.
- If bet, opponent may fold (`f`) or call (`c`). Higher card wins at showdown.

**Game tree:** 4 non-terminal histories, 5 terminal histories, 12 infoset nodes.

**Infoset key format:** `"<card>:<history>"` — e.g. `"K:cb"` (holds King, history is check-then-bet).

```bash
PYTHONPATH=. python kuhn/kuhn_poker.py   # sanity checks
```

---

## Phase 2 — CFR on Kuhn Poker

```bash
PYTHONPATH=. python kuhn/cfr.py          # trains 10,000 iterations, prints strategy
python -m pytest tests/ -v               # 124 tests passing
```

### Results after 10,000 iterations

> These are the **solver-produced** values from an actual training run — compare them against the analytical Nash Equilibrium Reference table above.

| Infoset | Strategy | Interpretation |
|---|---|---|
| `K:` | bet 63% | King bets for value |
| `Q:` | check 100% | Queen never opens |
| `J:` | bet 21% | Jack bluffs occasionally |
| `K:b` | call 100% | King always calls a bet |
| `Q:b` | call 33% | Queen calls 1/3 to prevent bluffs |
| `J:b` | fold 100% | Jack folds to a bet |
| `K:c` | bet 100% | King always bets after a check |
| `J:c` | bet 34% | Jack bluffs after opponent checks |
| `Q:cb` | call 54% | Queen calls check-bet often enough to deter bluffs |

**Game value:** −1/18 ≈ −0.0556 chips per hand for Player 1.

---

## Phase 3 — Exploitability Analysis

```bash
PYTHONPATH=. python kuhn/analysis.py     # trains + generates plots
```

### Convergence

![Kuhn Convergence](kuhn/plots/convergence.png)

### Strategy Heatmap

![Kuhn Strategy Heatmap](kuhn/plots/strategy_heatmap.png)

---

## Phase 4 — Leduc Poker

Leduc Poker (Southey et al., 2005) is the standard two-round benchmark with a community card.

**Rules:**
- Deck: J, J, Q, Q, K, K (6 cards, two of each rank)
- Both ante 1 chip. Round 1 bet size = 2, max 1 raise.
- Community card dealt face-up between rounds (chance node).
- Round 2 bet size = 4, max 1 raise.
- Showdown: pair (private matches community) beats non-pair; otherwise higher rank wins.

```bash
PYTHONPATH=. python leduc/leduc_poker.py   # sanity checks
PYTHONPATH=. python leduc/cfr.py           # trains 1,000 iterations
PYTHONPATH=. python leduc/analysis.py      # trains + generates 3 plots
```

### Convergence

![Leduc Convergence](leduc/plots/convergence.png)

### Strategy by Round

![Leduc Strategy by Round](leduc/plots/strategy_by_round.png)

### Community Card Effect

![Leduc Community Card Effect](leduc/plots/community_card_effect.png)

---

## Phase 5 — FastAPI Backend

REST API wrapping the CFR solver, serving trained GTO strategies for Kuhn and Leduc Poker.

### Running the server

```bash
pip install -r requirements.txt
PYTHONPATH=. python backend/run.py
```

Server starts at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/train` | Train a CFR solver for a game |
| `GET` | `/strategy/{game}` | Full strategy for all infosets |
| `GET` | `/strategy/{game}/{infoset}` | Strategy for a single infoset |
| `GET` | `/exploitability/{game}` | Exploitability of the trained strategy |
| `GET` | `/ev/{game}` | Final expected value from training |
| `GET` | `/health` | Health check + list of trained games |

### Example usage

```bash
# Train Kuhn Poker (10,000 iterations)
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"game": "kuhn", "iterations": 10000}'

# Get full strategy
curl http://localhost:8000/strategy/kuhn

# Query a single infoset (URL-encode the colon)
curl http://localhost:8000/strategy/kuhn/K%3A

# Get exploitability
curl http://localhost:8000/exploitability/kuhn

# Get EV
curl http://localhost:8000/ev/kuhn

# Health check
curl http://localhost:8000/health
```

---

## Phase 6 — React Frontend

An interactive single-page app for exploring the solved GTO strategies, built with React + TypeScript + Tailwind CSS (Vite).

**Features:**
- **Interactive card selector** — pick your private card (J / Q / K) and, in Leduc, the community card once Round 1 closes. Cards render as playing cards.
- **Action history builder** — click through the game tree with only the legal actions surfaced at each step (Check / Bet / Call / Raise / Fold). A breadcrumb shows the readable history and Leduc round transitions.
- **GTO strategy bars** — the solver's action probabilities for the selected infoset, drawn as colored horizontal bars with percentage labels.
- **Live stats** — nodes trained, game value (EV), and exploitability, with a color-coded exploitability badge (green < 0.01, yellow < 0.05, red otherwise).

The app talks to the FastAPI backend: it triggers training, fetches the full strategy, and looks up infoset keys client-side as you build a hand.

### Running the frontend

```bash
cd frontend
npm install
npm run dev   # opens http://localhost:5173
```

Make sure the backend is running at `http://localhost:8000` first (see Phase 5). The API base URL is configured via `VITE_API_URL` in `frontend/.env`.

---

## Running Tests

```bash
python -m pytest tests/ -v    # 124 tests, all passing
```

---

## Roadmap

- [x] Phase 1 — Kuhn Poker game environment (34 tests)
- [x] Phase 2 — Vanilla CFR following Zinkevich et al. (2007) (18 tests)
- [x] Phase 3 — Exploitability analysis and convergence plots
- [x] Phase 4 — Leduc Poker (two betting rounds, community card, chance node)
- [x] Phase 5 — FastAPI backend serving GTO strategies
- [x] Phase 6 — React + Tailwind frontend with interactive strategy explorer

---

## Deployment

### Backend → Render

1. Push the repo to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect the GitHub repo
4. Render auto-detects `render.yaml` — settings are pre-configured:
   - Build: `pip install -r requirements.txt`
   - Start: `PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. After deploy, update `frontend/.env.production` with the live Render URL

### Frontend → Vercel

1. Create a new project on [vercel.com](https://vercel.com)
2. Set the **Root Directory** to `frontend`
3. Framework preset: **Vite**
4. Set environment variable: `VITE_API_URL` = your Render backend URL
5. Deploy — Vercel auto-builds with `npm run build`
6. The `vercel.json` handles SPA routing (all paths → `index.html`)

---

## References

Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). **Regret Minimization in Games with Incomplete Information.** *Advances in Neural Information Processing Systems 20 (NeurIPS 2007).* https://proceedings.neurips.cc/paper/2007/file/08d98638c6a1e74d5d7506fd2fe68c53-Paper.pdf

Kuhn, H. W. (1950). A simplified two-person poker. In H. W. Kuhn & A. W. Tucker (Eds.), *Contributions to the Theory of Games*, Vol. 1, pp. 97–103. Princeton University Press.

Southey, F., Bowling, M., Larson, B., Piccione, C., Burch, N., Billings, D., & Rayner, C. (2005). **Bayes' Bluff: Opponent Modelling in Poker.** *Proceedings of the 21st Conference on Uncertainty in Artificial Intelligence (UAI 2005).*
