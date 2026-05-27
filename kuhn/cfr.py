"""
kuhn/cfr.py
───────────
Counterfactual Regret Minimization (CFR) for Kuhn Poker.

Reference:
    Zinkevich et al. (2007) "Regret Minimization in Games with Incomplete
    Information." NeurIPS 2007.

    This implementation follows Algorithm 1 from the paper as closely as
    possible. Variable names and notation match the paper where practical:
        - i           : the current player (0 or 1, paper uses 1 or 2)
        - π (pi)      : reach probability (called p0, p1 in code)
        - π^{-i}      : counterfactual reach (opponent's reach probability)
        - σ (sigma)   : the current strategy (mixed action distribution)
        - R^T_i       : cumulative regret for player i (regret_sum in code)
        - σ̄ (sigma bar) : average strategy (strategy_sum in code)

Architecture — two-pass design per iteration:
    The key correctness requirement from Zinkevich et al. is that all nodes
    use the SAME strategy profile within one iteration. If we update regrets
    mid-sweep, later deals in the same iteration see a different strategy than
    earlier deals, making each iteration order-dependent.

    We solve this with a strict two-pass design:

        Pass 1 — cfr_value():
            Traverse the game tree READ-ONLY with respect to regrets.
            Compute and RETURN counterfactual values and regret updates,
            but do NOT write to any node yet.

        Pass 2 — accumulate():
            After all deals are processed, apply the collected updates
            to regret_sum and strategy_sum in one atomic step.

    This guarantees all 6 deals in one iteration see the same strategy.
"""

import numpy as np
from kuhn.kuhn_poker import (
    KuhnPoker,
    is_terminal, whose_turn, get_legal_actions,
    terminal_payoff, infoset_key,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Node class
# ─────────────────────────────────────────────────────────────────────────────

class Node:
    """
    One information set in the CFR algorithm.

    Stores cumulative regrets and strategy sums across all iterations.
    After training, get_average_strategy() returns the GTO probabilities.

    Attributes
    ----------
    infoset      : str       — e.g. "K:cb"
    actions      : list[str] — legal actions here, e.g. ["f", "c"]
    n_actions    : int
    regret_sum   : np.array  — R^T_i(I, a) from Zinkevich et al. eq. 5
    strategy_sum : np.array  — accumulator for σ̄^T_i(I, a) from eq. 7
    """

    def __init__(self, infoset: str, actions: list[str]):
        self.infoset      = infoset
        self.actions      = actions
        self.n_actions    = len(actions)
        self.regret_sum   = np.zeros(self.n_actions)
        self.strategy_sum = np.zeros(self.n_actions)

    def get_strategy(self) -> np.ndarray:
        """
        Current strategy via regret matching (read-only — does not update state).

        σ(I, a) = R^+(I, a) / Σ_b R^+(I, b)

        If all regrets ≤ 0, play uniformly. This is correct because with no
        positive regret there is no evidence to prefer any action.
        """
        positive_regrets = np.maximum(self.regret_sum, 0)
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        return np.ones(self.n_actions) / self.n_actions

    def get_average_strategy(self) -> dict[str, float]:
        """
        Average strategy σ̄^T_i — this is what converges to Nash, not
        the current strategy. Call after training is complete.
        """
        total = self.strategy_sum.sum()
        avg = self.strategy_sum / total if total > 0 else np.ones(self.n_actions) / self.n_actions
        return {action: float(avg[i]) for i, action in enumerate(self.actions)}

    def __repr__(self) -> str:
        avg = self.get_average_strategy()
        parts = ", ".join(f"{a}={p:.3f}" for a, p in avg.items())
        return f"Node({self.infoset}: {parts})"


# ─────────────────────────────────────────────────────────────────────────────
#  CFRTrainer
# ─────────────────────────────────────────────────────────────────────────────

class CFRTrainer:
    """
    Vanilla CFR on Kuhn Poker — Zinkevich et al. Algorithm 1.

    Two-pass design ensures all deals in one iteration use the same
    frozen strategy profile. See module docstring for details.

    Usage:
        trainer = CFRTrainer()
        ev_history = trainer.train(iterations=10_000)
        strategy   = trainer.get_strategy()
    """

    def __init__(self):
        self.game  = KuhnPoker()
        self.nodes: dict[str, Node] = {}

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, iterations: int = 10_000) -> list[float]:
        """
        Run CFR for `iterations` iterations.

        Each iteration:
          1. Freeze the current strategy profile (snapshot of all node regrets).
          2. For each of the 6 possible deals, run cfr_value() to collect
             counterfactual regret updates — without writing to nodes yet.
          3. Sum the per-deal updates and apply them all at once (accumulate).
          4. Evaluate the current average strategy and record its EV.

        Linear weighting: strategy_sum at iteration t is weighted by t so that
        later (more accurate) iterations dominate the average. This improves
        convergence speed without changing theoretical guarantees.

        Returns
        -------
        list[float]
            Player 1's expected value of the average strategy after each
            iteration. Converges to KUHN_GAME_VALUE = -1/18 ≈ -0.0556.
        """
        ev_history = []

        for t in range(1, iterations + 1):

            # ── Step 1: freeze strategy profile ──────────────────────────────
            # All nodes use this snapshot for the entire iteration.
            # New nodes discovered mid-iteration default to uniform.
            strategy_profile: dict[str, np.ndarray] = {
                key: node.get_strategy()
                for key, node in self.nodes.items()
            }

            # ── Step 2: collect updates across all deals ──────────────────────
            # pending_updates[key] accumulates (regret_delta, strategy_delta)
            # across all 6 deals before anything is written to nodes.
            pending_updates: dict[str, list] = {}

            for p1_card, p2_card in self.game.get_all_deals():
                self._collect_updates(
                    history          = "",
                    p1_card          = p1_card,
                    p2_card          = p2_card,
                    p0               = 1.0,
                    p1               = 1.0,
                    t                = t,
                    strategy_profile = strategy_profile,
                    pending_updates  = pending_updates,
                )

            # ── Step 3: apply all updates atomically ──────────────────────────
            for key, (regret_delta, strategy_delta) in pending_updates.items():
                self.nodes[key].regret_sum   += regret_delta
                self.nodes[key].strategy_sum += strategy_delta

            # ── Step 4: record EV of average strategy ─────────────────────────
            ev_history.append(self._evaluate_average_strategy())

        return ev_history

    # ── Pass 1: collect updates (read-only on nodes) ──────────────────────────

    def _collect_updates(
        self,
        history          : str,
        p1_card          : int,
        p2_card          : int,
        p0               : float,
        p1               : float,
        t                : int,
        strategy_profile : dict[str, np.ndarray],
        pending_updates  : dict[str, list],
    ) -> float:
        """
        Traverse the game tree and collect regret + strategy updates.

        DOES NOT write to any node. Returns Player 1's counterfactual value
        at this node so the caller can compute regrets above.

        All recursive calls use strategy_profile (frozen at start of iteration)
        so every node in the tree uses the same strategy, regardless of the
        order deals are processed.
        """

        # Base case
        if is_terminal(history):
            return terminal_payoff(history, p1_card, p2_card)

        player  = whose_turn(history)
        card    = p1_card if player == 0 else p2_card
        key     = infoset_key(card, history)
        actions = get_legal_actions(history)

        # Ensure node exists
        if key not in self.nodes:
            self.nodes[key] = Node(key, actions)

        # Use frozen strategy — never reads live regrets mid-iteration
        if key not in strategy_profile:
            strategy_profile[key] = self.nodes[key].get_strategy()
        strategy = strategy_profile[key]

        # Recurse over actions
        action_values = np.zeros(len(actions))
        for i, action in enumerate(actions):
            next_history = history + action
            child_value  = self._collect_updates(
                next_history, p1_card, p2_card,
                p0 = p0 * strategy[i] if player == 0 else p0,
                p1 = p1 * strategy[i] if player == 1 else p1,
                t  = t,
                strategy_profile = strategy_profile,
                pending_updates  = pending_updates,
            )
            # action_values is always from current player's perspective
            action_values[i] = child_value if player == 0 else -child_value

        node_value      = float(np.dot(strategy, action_values))
        opponent_reach  = p1 if player == 0 else p0
        reach_prob      = p0 if player == 0 else p1

        # Compute deltas for this deal — do not apply yet
        regret_delta   = opponent_reach * (action_values - node_value)
        strategy_delta = t * reach_prob * strategy

        # Accumulate into pending_updates (summed across deals)
        if key not in pending_updates:
            pending_updates[key] = [
                np.zeros(len(actions)),  # regret accumulator
                np.zeros(len(actions)),  # strategy accumulator
            ]
        pending_updates[key][0] += regret_delta
        pending_updates[key][1] += strategy_delta

        # Always return from Player 1's perspective
        return node_value if player == 0 else -node_value

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate_average_strategy(self) -> float:
        """Player 1's EV under the current average strategy."""
        strategy = self.get_strategy()
        total    = 0.0
        for p1_card, p2_card in self.game.get_all_deals():
            total += self._eval_state("", p1_card, p2_card, strategy)
        return total / 6

    def _eval_state(
        self,
        history  : str,
        p1_card  : int,
        p2_card  : int,
        strategy : dict[str, dict[str, float]],
    ) -> float:
        """Recursive expected-value computation for a fixed strategy profile."""
        if is_terminal(history):
            return terminal_payoff(history, p1_card, p2_card)

        player  = whose_turn(history)
        card    = p1_card if player == 0 else p2_card
        key     = infoset_key(card, history)
        actions = get_legal_actions(history)

        return sum(
            strategy[key][a] * self._eval_state(history + a, p1_card, p2_card, strategy)
            for a in actions
        )

    # ── Results ───────────────────────────────────────────────────────────────

    def get_strategy(self) -> dict[str, dict[str, float]]:
        """Average strategy for all infosets — the Nash approximation."""
        return {key: node.get_average_strategy() for key, node in self.nodes.items()}

    def print_strategy(self) -> None:
        """Pretty-print the learned strategy grouped by player."""
        strategy = self.get_strategy()

        print("\n" + "=" * 55)
        print("Learned Strategy (Average over all iterations)")
        print("=" * 55)

        print("\nPlayer 1:")
        for key in sorted(k for k in strategy if k.endswith(":")):
            self._print_infoset(key, strategy[key])
        for key in sorted(k for k in strategy if k.endswith("cb")):
            self._print_infoset(key, strategy[key])

        print("\nPlayer 2:")
        for key in sorted(k for k in strategy if k.endswith(":b")):
            self._print_infoset(key, strategy[key])
        for key in sorted(k for k in strategy if k.endswith(":c")):
            self._print_infoset(key, strategy[key])

    def _print_infoset(self, key: str, action_probs: dict) -> None:
        parts = ", ".join(f"{a}={p:.3f}" for a, p in action_probs.items())
        print(f"  {key:<10} {parts}")


# ─────────────────────────────────────────────────────────────────────────────
#  Known Kuhn Nash Equilibrium — used for validation in tests
#
#  From Kuhn (1950) and Zinkevich et al. (2007):
#    Player 1: Jack bluffs 1/3, Queen always checks, King always bets
#    Player 2: Jack always folds, Queen calls 1/3, King always calls
#    Game value for Player 1: -1/18 ≈ -0.0556
# ─────────────────────────────────────────────────────────────────────────────

KUHN_NASH_EQUILIBRIUM = {
    "J:":   {"c": 2/3, "b": 1/3},
    "Q:":   {"c": 1.0, "b": 0.0},
    "K:":   {"c": 0.0, "b": 1.0},
    "J:cb": {"f": 1.0, "c": 0.0},
    "Q:cb": {"f": 2/3, "c": 1/3},
    "K:cb": {"f": 0.0, "c": 1.0},
    "J:b":  {"f": 1.0, "c": 0.0},
    "Q:b":  {"f": 2/3, "c": 1/3},
    "K:b":  {"f": 0.0, "c": 1.0},
    "J:c":  {"c": 1.0, "b": 0.0},
    "Q:c":  {"c": 1.0, "b": 0.0},
    "K:c":  {"c": 0.0, "b": 1.0},
}

KUHN_GAME_VALUE = -1 / 18


# ─────────────────────────────────────────────────────────────────────────────
#  Run directly to train and see results
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Training CFR on Kuhn Poker...")
    print("Following Zinkevich et al. (2007) Algorithm 1\n")

    trainer    = CFRTrainer()
    ev_history = trainer.train(iterations=10_000)

    trainer.print_strategy()

    print(f"\nFinal expected value (P1): {ev_history[-1]:.6f}")
    print(f"Known Nash game value:     {KUHN_GAME_VALUE:.6f}")
    print(f"Difference:                {abs(ev_history[-1] - KUHN_GAME_VALUE):.6f}")