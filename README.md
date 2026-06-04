# CFR Poker Solver

Counterfactual Regret Minimization (CFR) implemented from scratch in Python, progressing from Kuhn Poker through Leduc Poker to Heads-Up Texas Hold'em. Final product is a deployed interactive web app where you can query GTO strategies at any decision point.

---

## Live Demo

> *Coming in Phase 7 — React frontend + FastAPI backend, deployed on Vercel and Render*

---

## What is CFR?

Counterfactual Regret Minimization is the algorithm behind modern poker solvers (PioSOLVER, GTO+, Solver used by GGPoker). It was introduced by Zinkevich, Johanson, Bowling, and Piccione in their 2007 NeurIPS paper:

> Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). **Regret Minimization in Games with Incomplete Information.** *Advances in Neural Information Processing Systems 20 (NeurIPS 2007).*

The core idea: play a game thousands of times, track how much you *regret* not having taken a different action at each decision point, and gradually shift your strategy toward actions you regret not taking. The theoretical guarantee (Theorem 3 in the paper) is that the **average strategy** over all iterations converges to a Nash equilibrium — a strategy neither player can improve by deviating.

---

## The Algorithm

### Notation (following Zinkevich et al.)

| Symbol | Meaning |
|---|---|
| `I` | Information set — all game states a player cannot distinguish |
| `σ(I, a)` | Strategy: probability of taking action `a` at infoset `I` |
| `π^σ(h)` | Reach probability of history `h` under strategy `σ` |
| `π^σ_{-i}(h)` | Counterfactual reach: opponent's contribution to reach prob |
| `R^T_i(I, a)` | Cumulative counterfactual regret for action `a` at infoset `I` |
| `σ̄^T_i` | Average strategy over `T` iterations — converges to Nash |

### Regret Matching (the core update rule)

At each infoset, the current strategy is derived from accumulated regrets:

```
σ(I, a) = R⁺(I, a) / Σ_b R⁺(I, b)
```

where `R⁺(I, a) = max(R(I, a), 0)`. If all regrets are non-positive, play uniformly.

### Counterfactual Regret Update (Equation 5 in the paper)

After each traversal, regrets are updated as:

```
R^{T+1}(I, a) += π^σ_{-i}(I) · (v_σ(I, a) - v_σ(I))
```

The opponent's reach probability `π^σ_{-i}` weights the regret — this is the "counterfactual" part. We ask: *if I had always reached this infoset, how much would I regret not taking action `a`?*

### Two-Pass Design (correctness requirement)

A subtle but critical implementation detail: all nodes must use the **same strategy profile** within a single iteration. Updating regrets mid-sweep causes later deals to see a different strategy than earlier ones, making each iteration order-dependent.

This implementation uses a strict two-pass design per iteration:
1. **Freeze** the current strategy profile as a snapshot
2. **Collect** all regret and strategy updates across all deals without writing to nodes
3. **Apply** all updates atomically after the full sweep

This matches the simultaneous-update semantics described in Zinkevich et al.

---

## Project Structure

```
cfr-poker/
├── kuhn/
│   ├── kuhn_poker.py     # Game environment: rules, payoffs, infoset keys
│   ├── cfr.py            # CFR algorithm: Node class, CFRTrainer, Nash reference
│   ├── analysis.py       # Exploitability computation and convergence plots
│   └── plots/
│       ├── convergence.png
│       └── strategy_heatmap.png
├── leduc/                # Phase 4 — coming soon
├── holdem/               # Phase 5 — coming soon
├── backend/              # Phase 6 — FastAPI REST API
├── tests/
│   ├── test_kuhn_poker.py    # 34 tests — game environment
│   └── test_cfr_trainer.py   # 21 tests — CFR algorithm and convergence
└── README.md
```

---

## Phase 1 — Kuhn Poker Environment

Kuhn Poker is a 3-card, 2-player poker variant introduced by Harold Kuhn in 1950. It is the simplest non-trivial poker game and the standard benchmark for testing game-theoretic algorithms.

**Rules:**
- Deck: Jack (J), Queen (Q), King (K). Each player receives one card; one is burned.
- Both players ante 1 chip. One betting round: check or bet (1 chip).
- If bet, opponent may fold or call. Higher card wins at showdown.

**Information sets:** A player knows their own card and the public action history, but not the opponent's card. The infoset key format used throughout this project is `"<card>:<history>"` — for example, `"K:cb"` means the player holds a King and the action history is check-then-bet.

**Game tree:** 4 non-terminal histories, 5 terminal histories, 12 total infoset nodes.

---

## Phase 2 — CFR on Kuhn Poker

### Results after 10,000 iterations

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
| `Q:cb` | call 54% | Queen calls check-bet often enough |

**Game value:** −1/18 ≈ −0.0556 chips per hand for Player 1. Acting first in Kuhn Poker is a slight *disadvantage* — your bet reveals information about your hand strength without gaining enough in return.

### Convergence

![CFR Convergence](kuhn/plots/convergence.png)

Exploitability (how much an adversary could gain by deviating from the strategy) converges toward zero as iterations increase, consistent with the O(1/√T) bound proven in Theorem 3 of Zinkevich et al.

### Strategy Heatmap

![Strategy Heatmap](kuhn/plots/strategy_heatmap.png)

---

## Phase 3 — Validation and Analysis

**Exploitability** measures how far a strategy is from Nash equilibrium:

```
e(σ) = (BR₁ value − Nash value) + (BR₂ value − Nash value)
```

where BR_i is the best-response value for player i against the opponent's fixed strategy. At Nash equilibrium, `e(σ) = 0`.

**Key validation result:** The stored Nash equilibrium has exploitability < 0.003, confirming it is within numerical tolerance of a true Nash equilibrium. CFR exploitability decreases monotonically with iterations.

**Important finding during development:** Exploitability must be computed at the *infoset level*, not the deal level. A player cannot condition their action on the opponent's private card — their best response must be the same across all deals consistent with a given infoset. Averaging action values weighted by opponent reach probability across deals before selecting the best action is the correct approach.

---

## Setup

```bash
git clone https://github.com/RyanArpin/cfr-poker.git
cd cfr-poker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Run the solver:**
```bash
PYTHONPATH=. python kuhn/cfr.py
```

**Generate convergence plots:**
```bash
PYTHONPATH=. python kuhn/analysis.py
```

**Run tests:**
```bash
python -m pytest tests/ -v
```

---

## Roadmap

- [x] Phase 1 — Kuhn Poker game environment
- [x] Phase 2 — Vanilla CFR following Zinkevich et al. (2007)
- [x] Phase 3 — Exploitability analysis and convergence plots
- [ ] Phase 4 — Leduc Poker (two betting rounds, community card)
- [ ] Phase 5 — Heads-Up Texas Hold'em with Monte Carlo CFR
- [ ] Phase 6 — FastAPI backend serving GTO strategies
- [ ] Phase 7 — React + Tailwind frontend with interactive card selection

---

## References

Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007). Regret Minimization in Games with Incomplete Information. *Advances in Neural Information Processing Systems 20.* https://proceedings.neurips.cc/paper/2007/file/08d98638c6a1e74d5d7506fd2fe68c53-Paper.pdf

Kuhn, H. W. (1950). A simplified two-person poker. In H. W. Kuhn & A. W. Tucker (Eds.), *Contributions to the Theory of Games*, Vol. 1, pp. 97–103. Princeton University Press.