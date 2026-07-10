"""
holdem/cfr.py
─────────────
External Sampling Monte Carlo CFR (MCCFR) for Heads-Up Texas Hold'em.

Reference:
    Lanctot et al. (2009) "Monte Carlo Sampling for Regret Minimization in
    Extensive Games." NeurIPS 2009.

Why Monte Carlo CFR:
    Hold'em (even with card abstraction, 8 buckets) has far more infosets
    than Leduc.  Vanilla CFR iterates over the full tree on every pass —
    feasible for Kuhn (12 nodes) and Leduc (~288 nodes), but too slow here.

    External Sampling MCCFR samples one traversal per iteration:
      • Traversing player  — iterate over ALL their actions at each node
        (exactly like vanilla CFR).
      • Opponent and chance nodes — SAMPLE a single action / card deal.

    Cost drops from O(|tree|) to O(depth) per iteration.  The average
    strategy still converges to Nash — proven in Lanctot et al. Theorem 2.

Street transition model:
    History strings use "|" to separate streets:
        0 pipes  →  preflop
        1 pipe   →  flop
        2 pipes  →  turn
        3 pipes  →  river

    When the current street's action sequence is complete (matches a
    closed-no-fold pattern like "cc", "bk", "kc", …) AND we are not yet
    on the river, we advance by appending "|" to the history and recursing.
    This is a single clean step — NOT a mid-recursion branch that can loop.

    The traversal always passes the correct `street` index alongside the
    history, so bucket evaluation is consistent throughout.

Node class:
    Identical to kuhn/cfr.py — imported directly.
"""

from __future__ import annotations
import random
import numpy as np

from kuhn.cfr import Node
from holdem.holdem_poker import (
    HoldemPoker,
    is_terminal, whose_turn, get_legal_actions,
    street_is_over, advance_street,
    terminal_payoff, infoset_key,
    _current_street,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Target exploitability
# ─────────────────────────────────────────────────────────────────────────────

HOLDEM_EXPLOITABILITY_TARGET = 0.1   # chips per hand — approximate Nash target


class HoldemCFRTrainer:
    """
    External Sampling Monte Carlo CFR for Heads-Up Hold'em.

    Infoset keys: "<bucket>:<history>"  where bucket ∈ {0..7}.

    Usage
    -----
    trainer = HoldemCFRTrainer()
    ev_history = trainer.train(iterations=10_000)
    strategy   = trainer.get_strategy()
    """

    def __init__(self):
        self.game  = HoldemPoker()
        self.nodes: dict[str, Node] = {}

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, iterations: int = 10_000) -> list[float]:
        """
        Run External Sampling MCCFR for `iterations` iterations.

        Each iteration:
          1. Sample a fresh deal.
          2. Pre-compute both players' bucket at each of the 4 streets
             (once per deal, not once per node — avoids repeated hand eval).
          3. Traverse as P1 (traverser=0), updating P1 regrets.
          4. Traverse as P2 (traverser=1), updating P2 regrets.
          5. Record the P1-traversal value as the EV estimate.

        Linear weighting on strategy_sum (× iteration number t) matches
        the approach in kuhn/cfr.py and leduc/cfr.py.

        Returns
        -------
        list[float] — per-iteration EV estimates (noisy due to sampling).
        """
        ev_history = []

        for t in range(1, iterations + 1):

            deal = self.game.deal()

            # ── Pre-compute buckets for all streets (4 × hand-eval, once) ─────
            buckets: list[tuple[int, int]] = [
                self.game.get_buckets(deal, s) for s in range(4)
            ]

            # Freeze strategy snapshot
            strategy_profile: dict[str, np.ndarray] = {
                key: node.get_strategy()
                for key, node in self.nodes.items()
            }

            ev0 = self._traverse("", 0, buckets, 0, t, strategy_profile)
            self._traverse("", 1, buckets, 0, t, strategy_profile)
            ev_history.append(ev0)

        return ev_history

    # ── External sampling traversal ───────────────────────────────────────────

    def _traverse(
        self,
        history          : str,
        traverser        : int,
        buckets          : list[tuple[int, int]],
        street           : int,
        t                : int,
        strategy_profile : dict[str, np.ndarray],
    ) -> float:
        """
        External Sampling MCCFR traversal.

        Parameters
        ----------
        history          : current action history
        traverser        : 0 or 1 — which player's regrets we are updating
        buckets          : pre-computed [(p1_b, p2_b), ...] for streets 0–3
        street           : current betting street (0=preflop … 3=river)
        t                : iteration number (linear weighting)
        strategy_profile : frozen strategy snapshot

        Returns value from Player 1's perspective.
        """
        # ── Terminal ──────────────────────────────────────────────────────────
        if is_terminal(history):
            p1_b, p2_b = buckets[min(street, 3)]
            return terminal_payoff(history, p1_b, p2_b)

        # ── Street transition ─────────────────────────────────────────────────
        if street_is_over(history) and street < 3:
            return self._traverse(
                advance_street(history), traverser, buckets,
                street + 1, t, strategy_profile,
            )

        # ── Decision node ──────────────────────────────────────────────────────
        player  = whose_turn(history)
        p1_b, p2_b = buckets[street]
        bucket  = p1_b if player == 0 else p2_b
        key     = infoset_key(bucket, history)
        actions = get_legal_actions(history)

        if key not in self.nodes:
            self.nodes[key] = Node(key, actions)
        if key not in strategy_profile:
            strategy_profile[key] = self.nodes[key].get_strategy()
        strategy = strategy_profile[key]

        if player == traverser:
            # Iterate all actions for the traversing player
            action_values = np.zeros(len(actions))
            for i, action in enumerate(actions):
                val = self._traverse(
                    history + action, traverser, buckets,
                    street, t, strategy_profile,
                )
                action_values[i] = val if player == 0 else -val

            node_value = float(np.dot(strategy, action_values))
            self.nodes[key].regret_sum   += action_values - node_value
            self.nodes[key].strategy_sum += t * strategy
            return node_value if player == 0 else -node_value

        else:
            # Sample a single action for the opponent
            sampled = random.choices(actions, weights=strategy.tolist(), k=1)[0]
            return self._traverse(
                history + sampled, traverser, buckets,
                street, t, strategy_profile,
            )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def estimate_ev(self, n_samples: int = 500) -> float:
        """Estimate P1 EV by evaluating the average strategy on sampled deals."""
        strategy = self.get_strategy()
        total    = 0.0
        for _ in range(n_samples):
            deal    = self.game.deal()
            buckets = [self.game.get_buckets(deal, s) for s in range(4)]
            total  += self._eval_deal("", buckets, 0, strategy)
        return total / n_samples

    def _eval_deal(
        self,
        history  : str,
        buckets  : list[tuple[int, int]],
        street   : int,
        strategy : dict[str, dict[str, float]],
    ) -> float:
        """Expected value for one deal under a fixed strategy."""
        if is_terminal(history):
            p1_b, p2_b = buckets[min(street, 3)]
            return terminal_payoff(history, p1_b, p2_b)

        if street_is_over(history) and street < 3:
            return self._eval_deal(
                advance_street(history), buckets, street + 1, strategy
            )

        player  = whose_turn(history)
        p1_b, p2_b = buckets[street]
        bucket  = p1_b if player == 0 else p2_b
        key     = infoset_key(bucket, history)
        actions = get_legal_actions(history)

        probs = strategy.get(key, {a: 1.0 / len(actions) for a in actions})
        return sum(
            probs.get(a, 0.0) * self._eval_deal(
                history + a, buckets, street, strategy
            )
            for a in actions
        )

    # ── Results ───────────────────────────────────────────────────────────────

    def get_strategy(self) -> dict[str, dict[str, float]]:
        """Average strategy for all visited infosets."""
        return {key: node.get_average_strategy() for key, node in self.nodes.items()}

    def print_summary(self, ev_history: list[float]) -> None:
        """Print a brief training summary."""
        if not ev_history:
            return
        window     = min(200, len(ev_history))
        smoothed   = sum(ev_history[-window:]) / window
        print(f"\nIterations:   {len(ev_history):,}")
        print(f"Nodes:        {len(self.nodes):,}")
        print(f"EV (last {window} avg): {smoothed:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("Training External Sampling MCCFR on Hold'em...")
    print("Following Lanctot et al. (2009)\n")

    trainer = HoldemCFRTrainer()
    start   = time.time()
    ev_hist = trainer.train(iterations=10_000)
    elapsed = time.time() - start

    trainer.print_summary(ev_hist)
    print(f"Training time: {elapsed:.1f}s")

    ev_est = trainer.estimate_ev(500)
    print(f"EV estimate (500 samples): {ev_est:.4f}")
