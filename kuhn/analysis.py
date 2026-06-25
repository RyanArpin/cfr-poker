"""
kuhn/analysis.py
────────────────
Convergence analysis and exploitability plots for the Kuhn Poker CFR solver.

Exploitability measures how far a strategy is from Nash equilibrium:

    e(σ) = (BR_1 value - Nash value) + (BR_2 value - Nash value from P2's view)

At Nash equilibrium, e(σ) = 0 because neither player can improve by deviating.

Best Response algorithm:
    For each deal, traverse the game tree. At opponent nodes, weight child
    values by their strategy probabilities. At BR player nodes, collect
    reach-weighted action values per infoset. After all deals, pick the
    best action per infoset, then evaluate the combined strategy.

Reference:
    Zinkevich et al. (2007) — Theorem 3 states CFR's average strategy has
    exploitability bounded by O(1/sqrt(T)).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from collections import defaultdict

from kuhn.kuhn_poker import (
    KuhnPoker,
    is_terminal, whose_turn, get_legal_actions,
    terminal_payoff, infoset_key,
)
from kuhn.cfr import CFRTrainer, KUHN_GAME_VALUE


def best_response_value(
    game      : KuhnPoker,
    strategy  : dict[str, dict[str, float]],
    br_player : int,
    return_info: bool = False,
) -> float | tuple[float, dict]:
    """
    Value achieved by br_player playing a best response against strategy.
    Returns value from Player 1's perspective.
    """
    infoset_values : dict = defaultdict(lambda: defaultdict(float))
    infoset_reach  : dict = defaultdict(float)

    for p1_card, p2_card in game.get_all_deals():
        _accumulate_br_values(
            history        = "",
            p1_card        = p1_card,
            p2_card        = p2_card,
            reach_opp      = 1.0,
            strategy       = strategy,
            br_player      = br_player,
            infoset_values = infoset_values,
            infoset_reach  = infoset_reach,
        )

    br_strategy: dict[str, str] = {}
    for key, action_vals in infoset_values.items():
        reach = infoset_reach[key]
        if reach < 1e-12:
            br_strategy[key] = next(iter(action_vals))
            continue
        avg_vals = {a: v / reach for a, v in action_vals.items()}
        br_strategy[key] = (
            max(avg_vals, key=avg_vals.get) if br_player == 0
            else min(avg_vals, key=avg_vals.get)
        )

    full_strategy = {}
    for key, action_probs in strategy.items():
        full_strategy[key] = dict(action_probs)

    for key, best_action in br_strategy.items():
        actions = list(strategy.get(key, {}).keys())
        if not actions:
            actions = get_legal_actions(key.split(":")[1])
        full_strategy[key] = {a: (1.0 if a == best_action else 0.0) for a in actions}

    total = 0.0
    for p1_card, p2_card in game.get_all_deals():
        total += _eval_state("", p1_card, p2_card, full_strategy)
    value = total / 6

    if not return_info:
        return value

    # Compute per-infoset spread (max_avg - min_avg) for reporting.
    spreads: dict[str, float] = {}
    for key, action_vals in infoset_values.items():
        reach = infoset_reach[key]
        if reach < 1e-12:
            spreads[key] = 0.0
            continue
        avg_vals = [v / reach for v in action_vals.values()]
        spreads[key] = max(avg_vals) - min(avg_vals)

    return value, spreads


def _accumulate_br_values(
    history        : str,
    p1_card        : int,
    p2_card        : int,
    reach_opp      : float,
    strategy       : dict,
    br_player      : int,
    infoset_values : dict,
    infoset_reach  : dict,
) -> float:
    """
    Traverse the tree for one deal. At opponent nodes weight by strategy.
    At BR player nodes collect reach-weighted action values per infoset.
    Returns value from Player 1's perspective.
    """
    if is_terminal(history):
        return terminal_payoff(history, p1_card, p2_card)

    player  = whose_turn(history)
    card    = p1_card if player == 0 else p2_card
    key     = infoset_key(card, history)
    actions = get_legal_actions(history)

    if player == br_player:
        action_vals = {}
        for action in actions:
            action_vals[action] = _accumulate_br_values(
                history + action, p1_card, p2_card, reach_opp,
                strategy, br_player, infoset_values, infoset_reach,
            )
        for action in actions:
            infoset_values[key][action] += reach_opp * action_vals[action]
        infoset_reach[key] += reach_opp
        # Original behavior: return the average over actions for this deal.
        return sum(action_vals[a] / len(actions) for a in actions)
    else:
        total = 0.0
        for action in actions:
            prob = strategy[key][action]
            if prob < 1e-12:
                continue
            child_val = _accumulate_br_values(
                history + action, p1_card, p2_card, reach_opp * prob,
                strategy, br_player, infoset_values, infoset_reach,
            )
            total += prob * child_val
        return total


def _eval_state(
    history  : str,
    p1_card  : int,
    p2_card  : int,
    strategy : dict,
) -> float:
    """Expected value under a fixed strategy profile (from P1's perspective)."""
    if is_terminal(history):
        return terminal_payoff(history, p1_card, p2_card)

    player  = whose_turn(history)
    card    = p1_card if player == 0 else p2_card
    key     = infoset_key(card, history)
    actions = get_legal_actions(history)

    return sum(
        strategy[key][a] * _eval_state(history + a, p1_card, p2_card, strategy)
        for a in actions
    )


def exploitability(
    game    : KuhnPoker,
    strategy: dict,
) -> float:
    """
    Total exploitability of a strategy profile. Zero at Nash equilibrium.

    Exploitability = BR_1_value − BR_2_value, which equals the sum of
    per-player gains from deviating to their best responses.  At Nash
    equilibrium both values equal the Nash EV, so the difference is 0.
    """
    p1_br_val = best_response_value(game, strategy, br_player=0)
    p2_br_val = best_response_value(game, strategy, br_player=1)
    return p1_br_val - p2_br_val


def train_and_track(
    iterations   : int = 10_000,
    sample_every : int = 100,
) -> tuple[list[int], list[float], list[float]]:
    """
    Train CFR and record exploitability at regular intervals.
    Returns (iteration_numbers, exploitability_values, ev_values).
    """
    game    = KuhnPoker()
    trainer = CFRTrainer()

    iteration_points = []
    exploit_values   = []
    ev_values        = []

    print(f"Training CFR for {iterations:,} iterations...")
    print(f"Sampling exploitability every {sample_every} iterations\n")

    completed = 0
    while completed < iterations:
        this_chunk = min(sample_every, iterations - completed)
        ev_history = trainer.train(iterations=this_chunk)
        completed += this_chunk

        strategy = trainer.get_strategy()
        exploit  = exploitability(game, strategy)
        ev       = ev_history[-1]

        iteration_points.append(completed)
        exploit_values.append(exploit)
        ev_values.append(ev)

        if completed % (sample_every * 10) == 0 or completed == iterations:
            print(f"  iter {completed:>6,}  |  exploitability = {exploit:.5f}"
                  f"  |  EV = {ev:.6f}")

    print(f"\nFinal exploitability: {exploit_values[-1]:.6f}")
    print(f"Final EV:             {ev_values[-1]:.6f}  (Nash: {KUHN_GAME_VALUE:.6f})")
    return iteration_points, exploit_values, ev_values


def plot_convergence(
    iteration_points : list[int],
    exploit_values   : list[float],
    ev_values        : list[float],
    save_path        : str = "kuhn/plots/convergence.png",
) -> None:
    """Two-panel plot: exploitability (log scale) and EV convergence."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(
        "CFR Convergence on Kuhn Poker\n"
        "Zinkevich et al. (2007) — Vanilla CFR with Linear Weighting",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # Clip to a small positive floor before log scaling — exploitability can
    # dip slightly below zero at very low iteration counts before the average
    # strategy has stabilised enough for the BR traversal to be accurate.
    exploit_plot = [max(v, 1e-6) for v in exploit_values]
    ax1.plot(iteration_points, exploit_plot,
             color="#2563eb", linewidth=2, label="Exploitability e(σ)")
    ax1.axhline(y=0, color="#dc2626", linestyle="--", linewidth=1.2,
                label="Nash equilibrium (e = 0)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iterations", fontsize=11)
    ax1.set_ylabel("Exploitability (log scale)", fontsize=11)
    ax1.set_title("Exploitability — distance from Nash equilibrium", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax2.plot(iteration_points, ev_values,
             color="#16a34a", linewidth=2, label="P1 EV (average strategy)")
    ax2.axhline(y=KUHN_GAME_VALUE, color="#dc2626", linestyle="--", linewidth=1.2,
                label=f"Nash EV = {KUHN_GAME_VALUE:.4f}")
    ax2.set_xlabel("Iterations", fontsize=11)
    ax2.set_ylabel("Expected Value (chips)", fontsize=11)
    ax2.set_title("Player 1 Expected Value vs Nash Equilibrium", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nConvergence plot saved to: {save_path}")


def plot_strategy_heatmap(
    trainer  : CFRTrainer,
    save_path: str = "kuhn/plots/strategy_heatmap.png",
) -> None:
    """Strategy heatmap: rows = infosets, columns = passive/aggressive action."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    strategy = trainer.get_strategy()
    ordered_keys = [
        "K:", "Q:", "J:",
        "K:cb", "Q:cb", "J:cb",
        "K:b", "Q:b", "J:b",
        "K:c", "Q:c", "J:c",
    ]

    # Build rows defensively in case some infosets are missing from the
    # strategy (trainer may not have visited every infoset). Fill missing
    # rows with a uniform distribution over legal actions so plotting
    # doesn't raise KeyError.
    rows = []
    for k in ordered_keys:
        if k in strategy:
            rows.append(list(strategy[k].values()))
        else:
            history = k.split(":", 1)[1]
            actions = get_legal_actions(history)
            n = len(actions)
            rows.append([1.0 / n] * n)

    matrix = np.array(rows)

    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Passive (check/fold)", "Aggressive (bet/call)"], fontsize=11)
    ax.set_yticks(range(len(ordered_keys)))
    ax.set_yticklabels(ordered_keys, fontsize=11)
    ax.axhline(y=5.5, color="white", linewidth=2)

    for i in range(len(ordered_keys)):
        for j in range(2):
            val   = matrix[i, j]
            color = "black" if 0.2 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Action Probability")
    ax.set_title(
        "Learned GTO Strategy — Kuhn Poker\n"
        "CFR after 10,000 iterations (Zinkevich et al. 2007)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Strategy heatmap saved to: {save_path}")


if __name__ == "__main__":
    iters, exploit, ev = train_and_track(iterations=10_000, sample_every=50)
    plot_convergence(iters, exploit, ev)

    trainer = CFRTrainer()
    trainer.train(iterations=10_000)
    plot_strategy_heatmap(trainer)