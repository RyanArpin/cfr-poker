# CFR Poker Solver

Counterfactual Regret Minimization (CFR) implemented from scratch in Python, following Zinkevich et al. (2007), *"Regret Minimization in Games with Incomplete Information."* The project builds through two poker variants — Kuhn Poker and Leduc Poker — then wraps the solver in a REST API with an interactive React frontend.

*Student project — First year Honours Applied Mathematics, University of Waterloo.*

---

## Demo

**Kuhn Poker**

![Kuhn Poker demo](docs/kuhn-demo.gif)

**Leduc Poker**

![Leduc Poker demo](docs/leduc-demo.gif)

Train the solver, select a hand, and read the game-theory-optimal (GTO) strategy for every decision point.

---

## What is CFR?

CFR is the algorithm behind modern poker solvers (PioSOLVER, GTO+, etc.). The idea: play a game thousands of times, track how much you *regret* not having taken a different action at each decision point, and shift your strategy toward the actions you regret not taking. The average strategy over all iterations converges to a Nash equilibrium (Theorem 3, Zinkevich et al.).

**Regret matching:**

```
σ(I, a) = R⁺(I, a) / Σ_b R⁺(I, b)
```

where `R⁺ = max(R, 0)`. If all regrets are non-positive, play uniformly.

**Two-pass design (correctness requirement).** All nodes must see the same strategy profile within one iteration; updating regrets mid-sweep makes each iteration order-dependent. This implementation uses a strict two-pass approach: freeze the current strategy as a snapshot, collect all regret/strategy deltas across every deal without writing to nodes, then apply all updates atomically after the full sweep.

---

## Nash Equilibrium Reference

### Kuhn Poker — analytical equilibrium

Kuhn Poker has a *known closed-form* Nash equilibrium (Kuhn, 1950) — not a single strategy but a one-parameter family, parameterized by a bluff frequency `α ∈ [0, 1/3]`.

| Infoset | Meaning | Nash Strategy |
|---|---|---|
| `J:` | P1 opens with Jack | bet with prob `α` (≤ 1/3), else check |
| `Q:` | P1 opens with Queen | always check |
| `K:` | P1 opens with King | bet with prob `3α` |
| `J:b` | Jack facing a bet | always fold |
| `Q:b` | Queen facing a bet | call with prob 1/3 |
| `K:b` | King facing a bet | always call |
| `J:c` | Jack after opponent checks | bet with prob 1/3 |
| `Q:c` | Queen after opponent checks | always check |
| `K:c` | King after opponent checks | always bet |
| `Q:cb` | Queen facing check-then-bet | call with prob 1/3 |

The solver converges to one member of this family, and the game value is exactly **−1/18 ≈ −0.0556** for Player 1 — acting first is a slight disadvantage, a known counterintuitive result. The trained strategy matches this equilibrium to within an exploitability of **~0.005**.

### Leduc Poker — no closed form

Leduc has **no closed-form Nash equilibrium**: the two-round structure with a community card dealt between rounds makes it analytically intractable — which is exactly why an iterative solver is valuable. This implementation discovers **~288 information sets** and converges to a low-exploitability approximate equilibrium. The learned strategy shows the expected behaviour: aggression spikes when the private card pairs with the community card (see the community card effect plot).

---

## Project Structure

```
cfr-poker/
├── kuhn/                 # Kuhn Poker: environment, CFR, analysis + plots
├── leduc/                # Leduc Poker: environment, CFR, analysis + plots
├── backend/              # FastAPI REST API serving trained strategies
├── frontend/             # React + TypeScript + Tailwind strategy explorer
├── tests/                # 124 tests across game logic and CFR convergence
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

> Run scripts from the project root with `PYTHONPATH=.` to avoid import errors.

---

## Kuhn Poker

The simplest non-trivial poker game (Kuhn, 1950) and the standard CFR benchmark.

- Deck: Jack, Queen, King. Each player gets one card.
- Both ante 1 chip. One betting round: check (`c`) or bet (`b`).
- Facing a bet, a player may fold (`f`) or call (`c`). Higher card wins at showdown.
- Game tree: 12 information sets. Infoset key format: `"<card>:<history>"` (e.g. `"K:cb"`).

```bash
PYTHONPATH=. python kuhn/cfr.py        # train + print strategy
PYTHONPATH=. python kuhn/analysis.py   # train + generate plots
```

### Results after 10,000 iterations

These are the solver-produced values — compare against the analytical Nash table above.

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

Plots are saved to [`kuhn/plots/`](kuhn/plots/) — a convergence chart (`convergence.png`) and a strategy heatmap (`strategy_heatmap.png`).

---

## Leduc Poker

A two-round benchmark with a community card.

- Deck: J, J, Q, Q, K, K. Each player gets one private card.
- Both ante 1 chip. Round 1 bet size 2, Round 2 bet size 4, max one raise per round.
- A community card is dealt face-up between rounds (a chance node).
- Showdown: a pair (private matches community) beats a non-pair; otherwise higher rank wins.
- Infoset key format: `"<card>/<community_or_->:<history>"` (e.g. `"K/-:b"`, `"K/J:bk/J/c"`).

```bash
PYTHONPATH=. python leduc/cfr.py        # train
PYTHONPATH=. python leduc/analysis.py   # train + generate plots
```

Plots are saved to [`leduc/plots/`](leduc/plots/) — a convergence chart (`convergence.png`), a per-round strategy heatmap (`strategy_by_round.png`), and the community card effect plot (`community_card_effect.png`).

---

## Backend (FastAPI)

A REST API wrapping the solver, serving trained strategies for both games.

```bash
PYTHONPATH=. python backend/run.py   # serves at http://localhost:8000
```

Interactive API docs are available at `http://localhost:8000/docs`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/train` | Train a CFR solver for a game |
| `GET` | `/strategy/{game}` | Full strategy for all infosets |
| `GET` | `/strategy/{game}/{infoset}` | Strategy for a single infoset |
| `GET` | `/exploitability/{game}` | Exploitability of the trained strategy |
| `GET` | `/ev/{game}` | Final expected value from training |
| `GET` | `/health` | Health check + list of trained games |

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"game": "kuhn", "iterations": 10000}'

curl http://localhost:8000/strategy/kuhn
```

---

## Frontend (React)

An interactive single-page app for exploring the solved strategies, built with React + TypeScript + Tailwind CSS (Vite).

- **Card selector** — pick a private card (and, in Leduc, the community card once Round 1 closes).
- **History builder** — click through the game tree with only the legal actions surfaced at each step.
- **GTO strategy bars** — the solver's action probabilities for the selected infoset.
- **Live stats** — nodes trained, game value, and a color-coded exploitability badge.

```bash
cd frontend
npm install
npm run dev   # opens http://localhost:5173
```

The backend must be running at `http://localhost:8000` first. The API base URL is set via `VITE_API_URL` in `frontend/.env`.

---

## Tests

```bash
python -m pytest tests/ -v   # 124 tests, all passing
```

Coverage: card constants, terminal detection, turn order, legal actions, payoffs, infoset keys, deal enumeration, CFR convergence to the Nash game value, Nash strategy properties, and two-pass update atomicity.

---

## Reference

Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). *Regret Minimization in Games with Incomplete Information.* Advances in Neural Information Processing Systems 20 (NeurIPS 2007).
