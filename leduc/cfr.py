"""
leduc/cfr.py
────────────
Counterfactual Regret Minimization (CFR) for Leduc Poker.

Reference:
    Zinkevich et al. (2007) "Regret Minimization in Games with Incomplete
    Information." NeurIPS 2007.

Architecture — two-pass design per iteration (same as kuhn/cfr.py):
    The correctness requirement is that all nodes see the SAME strategy
    profile within one iteration.  We use a strict two-pass design:

        Pass 1 — _collect_updates():
            Traverse the game tree READ-ONLY with respect to regrets.
            Compute and RETURN counterfactual values and regret updates
            but do NOT write to any node yet.

            At a chance node (round 1 just ended, community card not yet
            dealt), iterate over all possible community cards weighted by
            their probability — exactly as kuhn/cfr.py iterates over all
            private deals weighted uniformly.  No regrets or strategy sums
            are updated at the chance node itself.

        Pass 2 — accumulate:
            After all deals are processed, apply the collected updates
            to regret_sum and strategy_sum in one atomic step.

    This guarantees all 30 private deals in one iteration see the same
    frozen strategy profile.

Dealing model note:
    get_all_private_deals() returns 30 ordered (p1_idx, p2_idx) pairs
    drawn from the 6-card indexed deck.  Iterating over all 30 uniformly
    correctly weights deal probabilities (same-rank pairs occur twice,
    giving them double the weight vs mixed-rank pairs — matching true
    probabilities 4/30 and 4/30 respectively).  Rank is recovered via
    DECK[index] before any game logic.

    The Node class is identical to kuhn/cfr.py; we import it directly.
"""

import numpy as np

from kuhn.cfr import Node                          # reuse Node unchanged
from leduc.leduc_poker import (
    LeducPoker, DECK,
    is_terminal, is_chance_node, whose_turn,
    get_legal_actions, terminal_payoff, infoset_key,
    CARD_NAMES,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Known Leduc Nash game value
#
#  Set to None until a well-converged run establishes the empirical value.
#  Leduc Poker does not have a closed-form Nash EV like Kuhn Poker does.
# ─────────────────────────────────────────────────────────────────────────────

LEDUC_GAME_VALUE = None   # updated empirically after training


class LeducCFRTrainer:
    """
    Vanilla CFR on Leduc Poker — Zinkevich et al. Algorithm 1.

    Two-pass design ensures all deals in one iteration use the same frozen
    strategy profile.  See module docstring for details.

    Usage
    -----
    trainer = LeducCFRTrainer()
    ev_history = trainer.train(iterations=1_000)
    strategy   = trainer.get_strategy()
    """

    def __init__(self):
        self.game  = LeducPoker()
        self.nodes: dict[str, Node] = {}

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, iterations: int = 1_000, track_ev: bool = True) -> list[float]:
        """
        Run CFR for `iterations` iterations.

        Each iteration:
          1. Freeze the current strategy profile.
          2. For each of the 30 possible private deals, run _collect_updates()
             to collect counterfactual regret updates — without writing to
             nodes yet.  Community card enumeration happens inside
             _collect_updates at the chance node.
          3. Apply all updates atomically.
          4. (Optional) Evaluate the current average strategy EV and record it.

        Linear weighting: strategy_sum at iteration t is weighted by t so
        later (more accurate) iterations dominate the average.  Matches
        the approach in kuhn/cfr.py.

        Parameters
        ----------
        iterations : int
            Number of CFR iterations to run.
        track_ev : bool
            When True (default), evaluate the average-strategy EV every
            iteration and record the full convergence history — this is the
            behaviour the __main__ block and tests rely on.  When False, skip
            the expensive per-iteration full-tree traversal entirely and
            compute the EV only once after the loop, returning a
            single-element list `[final_ev]`.  Use the fast path when only the
            final strategy matters (e.g. serving requests on constrained CPU).

        Returns
        -------
        list[float]
            Player 1's expected value of the average strategy after each
            iteration (track_ev=True), or a single-element list with the final
            EV (track_ev=False).
        """
        ev_history = []

        for t in range(1, iterations + 1):

            # ── Step 1: freeze strategy profile ──────────────────────────────
            strategy_profile: dict[str, np.ndarray] = {
                key: node.get_strategy()
                for key, node in self.nodes.items()
            }

            # ── Step 2: collect updates across all 30 private deals ───────────
            pending_updates: dict[str, list] = {}

            for p1_idx, p2_idx in self.game.get_all_private_deals():
                p1_rank = DECK[p1_idx]
                p2_rank = DECK[p2_idx]
                self._collect_updates(
                    history          = "",
                    p1_rank          = p1_rank,
                    p2_rank          = p2_rank,
                    p1_idx           = p1_idx,
                    p2_idx           = p2_idx,
                    community_rank   = None,
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

            # ── Step 4: record EV of average strategy (optional) ──────────────
            if track_ev:
                ev_history.append(self._evaluate_average_strategy())

        # Fast path: compute EV once after the loop instead of every iteration.
        if not track_ev:
            ev_history.append(self._evaluate_average_strategy())

        return ev_history

    # ── Pass 1: collect updates (read-only on nodes) ──────────────────────────

    def _collect_updates(
        self,
        history          : str,
        p1_rank          : int,
        p2_rank          : int,
        p1_idx           : int,
        p2_idx           : int,
        community_rank   : int | None,
        p0               : float,
        p1               : float,
        t                : int,
        strategy_profile : dict[str, np.ndarray],
        pending_updates  : dict[str, list],
    ) -> float:
        """
        Traverse the game tree and collect regret + strategy updates.

        DOES NOT write to any node.  Returns Player 1's counterfactual value
        at this node so the caller can compute regrets above.

        All recursive calls use strategy_profile (frozen at start of
        iteration) so every node uses the same strategy regardless of deal
        order.

        Chance node:
            When is_chance_node(history) is True the next step is not a
            player decision but a card draw.  We iterate over all possible
            community card ranks weighted by probability (using p1_idx and
            p2_idx to condition the remaining deck).  The chance node itself
            has no regret or strategy update — it returns a probability-
            weighted sum of child values.
        """

        # ── Base case: terminal ───────────────────────────────────────────────
        if is_terminal(history):
            # For R1-fold terminals, community_rank is irrelevant — pass 0.
            comm = community_rank if community_rank is not None else 0
            return terminal_payoff(history, p1_rank, p2_rank, comm)

        # ── Chance node: deal community card ─────────────────────────────────
        if is_chance_node(history):
            total_value = 0.0
            for c_rank, prob in self.game.get_community_cards(p1_idx, p2_idx):
                # history + "/X/" marks the start of round 2
                new_history = f"{history}/{CARD_NAMES[c_rank]}/"
                child_value = self._collect_updates(
                    history          = new_history,
                    p1_rank          = p1_rank,
                    p2_rank          = p2_rank,
                    p1_idx           = p1_idx,
                    p2_idx           = p2_idx,
                    community_rank   = c_rank,
                    p0               = p0,
                    p1               = p1,
                    t                = t,
                    strategy_profile = strategy_profile,
                    pending_updates  = pending_updates,
                )
                total_value += prob * child_value
            return total_value

        # ── Player decision node ──────────────────────────────────────────────
        player  = whose_turn(history)
        rank    = p1_rank if player == 0 else p2_rank
        key     = infoset_key(rank, community_rank, history)
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
            child_value = self._collect_updates(
                history + action, p1_rank, p2_rank, p1_idx, p2_idx,
                community_rank,
                p0 = p0 * strategy[i] if player == 0 else p0,
                p1 = p1 * strategy[i] if player == 1 else p1,
                t  = t,
                strategy_profile = strategy_profile,
                pending_updates  = pending_updates,
            )
            # action_values always from current player's perspective
            action_values[i] = child_value if player == 0 else -child_value

        node_value     = float(np.dot(strategy, action_values))
        opponent_reach = p1 if player == 0 else p0
        reach_prob     = p0 if player == 0 else p1

        # Compute deltas — do not apply yet
        regret_delta   = opponent_reach * (action_values - node_value)
        strategy_delta = t * reach_prob * strategy

        # Accumulate into pending_updates (summed across deals)
        if key not in pending_updates:
            pending_updates[key] = [
                np.zeros(len(actions)),
                np.zeros(len(actions)),
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
        deals    = self.game.get_all_private_deals()
        for p1_idx, p2_idx in deals:
            total += self._eval_state(
                history        = "",
                p1_rank        = DECK[p1_idx],
                p2_rank        = DECK[p2_idx],
                p1_idx         = p1_idx,
                p2_idx         = p2_idx,
                community_rank = None,
                strategy       = strategy,
            )
        return total / len(deals)

    def _eval_state(
        self,
        history        : str,
        p1_rank        : int,
        p2_rank        : int,
        p1_idx         : int,
        p2_idx         : int,
        community_rank : int | None,
        strategy       : dict[str, dict[str, float]],
    ) -> float:
        """Recursive expected-value computation for a fixed strategy profile."""
        if is_terminal(history):
            comm = community_rank if community_rank is not None else 0
            return terminal_payoff(history, p1_rank, p2_rank, comm)

        if is_chance_node(history):
            total = 0.0
            for c_rank, prob in self.game.get_community_cards(p1_idx, p2_idx):
                new_history = f"{history}/{CARD_NAMES[c_rank]}/"
                total += prob * self._eval_state(
                    new_history, p1_rank, p2_rank, p1_idx, p2_idx,
                    c_rank, strategy,
                )
            return total

        player  = whose_turn(history)
        rank    = p1_rank if player == 0 else p2_rank
        key     = infoset_key(rank, community_rank, history)
        actions = get_legal_actions(history)

        if key not in strategy:
            # Node not yet trained — play uniform
            probs = {a: 1.0 / len(actions) for a in actions}
        else:
            probs = strategy[key]

        return sum(
            probs.get(a, 0.0) * self._eval_state(
                history + a, p1_rank, p2_rank, p1_idx, p2_idx,
                community_rank, strategy,
            )
            for a in actions
        )

    # ── Results ───────────────────────────────────────────────────────────────

    def get_strategy(self) -> dict[str, dict[str, float]]:
        """Average strategy for all infosets — the Nash approximation."""
        return {key: node.get_average_strategy() for key, node in self.nodes.items()}

    def print_strategy(self) -> None:
        """Pretty-print the learned strategy."""
        strategy = self.get_strategy()
        print("\n" + "=" * 60)
        print("Leduc Poker — Learned Strategy (Average)")
        print("=" * 60)

        print("\nRound 1 (community not yet dealt):")
        r1_keys = sorted(k for k in strategy if "/-:" in k)
        for key in r1_keys:
            parts = ", ".join(f"{a}={p:.3f}" for a, p in strategy[key].items())
            print(f"  {key:<20} {parts}")

        print("\nRound 2 (community visible):")
        r2_keys = sorted(k for k in strategy if "/-:" not in k)
        for key in r2_keys:
            parts = ", ".join(f"{a}={p:.3f}" for a, p in strategy[key].items())
            print(f"  {key:<30} {parts}")

    def print_summary(self, ev_history: list[float]) -> None:
        """Print convergence summary."""
        print(f"\nIterations:         {len(ev_history)}")
        print(f"Final EV (P1):      {ev_history[-1]:.6f}")
        print(f"Infoset nodes:      {len(self.nodes)}")


# ─────────────────────────────────────────────────────────────────────────────
#  Run directly to train and see results
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("Training CFR on Leduc Poker...")
    print("Following Zinkevich et al. (2007)\n")

    trainer = LeducCFRTrainer()

    start = time.time()
    ev_history = trainer.train(iterations=1_000)
    elapsed = time.time() - start

    trainer.print_summary(ev_history)
    trainer.print_strategy()

    print(f"\nTraining time: {elapsed:.2f}s")
    print(f"EV at iter 100:  {ev_history[99]:.6f}")
    print(f"EV at iter 1000: {ev_history[-1]:.6f}")
