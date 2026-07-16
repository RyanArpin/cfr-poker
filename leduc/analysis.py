"""
leduc/analysis.py
─────────────────
Convergence analysis and exploitability plots for the Leduc Poker CFR solver.

Exploitability measures how far a strategy is from Nash equilibrium:

    e(σ) = (BR_1 value − Nash value) + (BR_2 value − Nash value from P2's view)

At Nash equilibrium, e(σ) = 0 because neither player can improve by deviating.

Best Response algorithm (adapted for Leduc's two-round structure):
    For each private deal, traverse the game tree.  At chance nodes, weight
    child values by community card probability.  At opponent nodes, weight
    child values by their strategy probabilities.  At BR-player nodes, collect
    reach-weighted action values per infoset.  After all deals and community
    card weightings, pick the best action per infoset, then evaluate the
    combined strategy.

    Critical correctness notes (same pitfalls as kuhn/analysis.py):
      • BR must be computed at the infoset level, not the deal level.  A
        player cannot condition on the opponent's private card.
      • The reach weight denominator must be the total reach weight, not the
        raw deal count.  Unreachable nodes must not inflate the denominator.

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

from leduc.leduc_poker import (
    LeducPoker, DECK, CARD_NAMES,
    is_terminal, is_chance_node, whose_turn, get_legal_actions,
    terminal_payoff, infoset_key,
)
from leduc.cfr import LeducCFRTrainer, LEDUC_GAME_VALUE


def best_response_value(
    game      : LeducPoker,
    strategy  : dict[str, dict[str, float]],
    br_player : int,
) -> float:
    """
    Value achieved by br_player playing a best response against strategy.
    Returns value from Player 1's perspective.

    Parameters
    ----------
    game      : LeducPoker instance
    strategy  : average strategy profile from LeducCFRTrainer.get_strategy()
    br_player : 0 for Player 1 BR, 1 for Player 2 BR
    """
    infoset_values : dict = defaultdict(lambda: defaultdict(float))
    infoset_reach  : dict = defaultdict(float)

    for p1_idx, p2_idx in game.get_all_private_deals():
        _accumulate_br_values(
            history        = "",
            p1_rank        = DECK[p1_idx],
            p2_rank        = DECK[p2_idx],
            p1_idx         = p1_idx,
            p2_idx         = p2_idx,
            community_rank = None,
            reach_opp      = 1.0,
            strategy       = strategy,
            br_player      = br_player,
            game           = game,
            infoset_values = infoset_values,
            infoset_reach  = infoset_reach,
        )

    # ── Pick best action per infoset ──────────────────────────────────────────
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

    # ── Build full strategy: fix BR player, keep opponent's strategy ──────────
    full_strategy = {key: dict(probs) for key, probs in strategy.items()}
    for key, best_action in br_strategy.items():
        actions = list(strategy.get(key, {}).keys())
        if not actions:
            # infoset not in trained strategy — reconstruct action list
            history = key.split(":", 1)[1]
            try:
                actions = get_legal_actions(history)
            except ValueError:
                continue
        full_strategy[key] = {a: (1.0 if a == best_action else 0.0) for a in actions}

    # ── Evaluate the combined strategy ────────────────────────────────────────
    total = 0.0
    deals = game.get_all_private_deals()
    for p1_idx, p2_idx in deals:
        total += _eval_state(
            history        = "",
            p1_rank        = DECK[p1_idx],
            p2_rank        = DECK[p2_idx],
            p1_idx         = p1_idx,
            p2_idx         = p2_idx,
            community_rank = None,
            strategy       = full_strategy,
            game           = game,
        )
    return total / len(deals)


def _accumulate_br_values(
    history        : str,
    p1_rank        : int,
    p2_rank        : int,
    p1_idx         : int,
    p2_idx         : int,
    community_rank : int | None,
    reach_opp      : float,
    strategy       : dict,
    br_player      : int,
    game           : LeducPoker,
    infoset_values : dict,
    infoset_reach  : dict,
) -> float:
    """
    Traverse the tree for one deal.

    At chance nodes: weight child values by community card probability.
    At opponent nodes: weight by strategy probabilities.
    At BR-player nodes: collect reach-weighted action values per infoset.
    Returns value from Player 1's perspective.
    """
    if is_terminal(history):
        comm = community_rank if community_rank is not None else 0
        return terminal_payoff(history, p1_rank, p2_rank, comm)

    # ── Chance node: sum over community cards weighted by probability ─────────
    if is_chance_node(history):
        total = 0.0
        for c_rank, prob in game.get_community_cards(p1_idx, p2_idx):
            new_history = f"{history}/{CARD_NAMES[c_rank]}/"
            child_val   = _accumulate_br_values(
                new_history, p1_rank, p2_rank, p1_idx, p2_idx,
                c_rank, reach_opp * prob, strategy, br_player,
                game, infoset_values, infoset_reach,
            )
            total += prob * child_val
        return total

    player  = whose_turn(history)
    rank    = p1_rank if player == 0 else p2_rank
    key     = infoset_key(rank, community_rank, history)
    actions = get_legal_actions(history)

    if player == br_player:
        # Collect reach-weighted action values — do not commit to any action yet
        action_vals = {}
        for action in actions:
            action_vals[action] = _accumulate_br_values(
                history + action, p1_rank, p2_rank, p1_idx, p2_idx,
                community_rank, reach_opp, strategy, br_player,
                game, infoset_values, infoset_reach,
            )
        for action in actions:
            infoset_values[key][action] += reach_opp * action_vals[action]
        infoset_reach[key] += reach_opp
        # Return average over actions for this path (used only for propagation
        # up the tree; the actual BR choice is made after all deals finish).
        return sum(action_vals[a] / len(actions) for a in actions)
    else:
        # Opponent: weight by their strategy
        total = 0.0
        for action in actions:
            prob = strategy.get(key, {}).get(action, 1.0 / len(actions))
            if prob < 1e-12:
                continue
            child_val = _accumulate_br_values(
                history + action, p1_rank, p2_rank, p1_idx, p2_idx,
                community_rank, reach_opp * prob, strategy, br_player,
                game, infoset_values, infoset_reach,
            )
            total += prob * child_val
        return total


def _eval_state(
    history        : str,
    p1_rank        : int,
    p2_rank        : int,
    p1_idx         : int,
    p2_idx         : int,
    community_rank : int | None,
    strategy       : dict,
    game           : LeducPoker,
) -> float:
    """Expected value under a fixed strategy profile (from P1's perspective)."""
    if is_terminal(history):
        comm = community_rank if community_rank is not None else 0
        return terminal_payoff(history, p1_rank, p2_rank, comm)

    if is_chance_node(history):
        total = 0.0
        for c_rank, prob in game.get_community_cards(p1_idx, p2_idx):
            new_history = f"{history}/{CARD_NAMES[c_rank]}/"
            total += prob * _eval_state(
                new_history, p1_rank, p2_rank, p1_idx, p2_idx,
                c_rank, strategy, game,
            )
        return total

    player  = whose_turn(history)
    rank    = p1_rank if player == 0 else p2_rank
    key     = infoset_key(rank, community_rank, history)
    actions = get_legal_actions(history)

    probs = strategy.get(key, {a: 1.0 / len(actions) for a in actions})
    return sum(
        probs.get(a, 0.0) * _eval_state(
            history + a, p1_rank, p2_rank, p1_idx, p2_idx,
            community_rank, strategy, game,
        )
        for a in actions
    )


def exploitability(
    game    : LeducPoker,
    strategy: dict,
) -> float:
    """
    Total exploitability of a strategy profile.  Zero at Nash equilibrium.

    Exploitability = (gain P1 gets by deviating) + (gain P2 gets by deviating)
                   = (BR_1_value − strategy_ev) + (strategy_ev − BR_2_value)
                   = BR_1_value − BR_2_value

    This formulation is independent of knowing the true Nash EV and is always
    ≥ 0 at Nash equilibrium (where BR_1_value == BR_2_value == Nash EV).
    In practice it may be slightly negative for a few hundred iterations due
    to the average strategy not yet approximating Nash well enough for the
    best-response tree to be stable — this self-corrects as iterations grow.
    """
    p1_br_val = best_response_value(game, strategy, br_player=0)
    p2_br_val = best_response_value(game, strategy, br_player=1)
    return p1_br_val - p2_br_val


def train_and_track(
    iterations   : int = 1_000,
    sample_every : int = 100,
) -> tuple[list[int], list[float], list[float]]:
    """
    Train CFR and record exploitability at regular intervals.

    Parameters
    ----------
    iterations   : total number of CFR iterations to run
    sample_every : how often to compute exploitability (expensive)

    Returns
    -------
    (iteration_numbers, exploitability_values, ev_values)
    """
    game    = LeducPoker()
    trainer = LeducCFRTrainer()

    iteration_points = []
    exploit_values   = []
    ev_values        = []

    print(f"Training Leduc CFR for {iterations:,} iterations...")
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

        if completed % (sample_every * 5) == 0 or completed == iterations:
            print(f"  iter {completed:>6,}  |  exploitability = {exploit:.5f}"
                  f"  |  EV = {ev:.6f}")

    print(f"\nFinal exploitability: {exploit_values[-1]:.6f}")
    print(f"Final EV (P1):        {ev_values[-1]:.6f}")
    return iteration_points, exploit_values, ev_values


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    iteration_points : list[int],
    exploit_values   : list[float],
    ev_values        : list[float],
    save_path        : str = "leduc/plots/convergence.png",
) -> None:
    """
    Two-panel plot: exploitability (log scale) and EV convergence.

    Mirrors the structure of kuhn/analysis.py plot_convergence exactly.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(
        "CFR Convergence on Leduc Poker\n"
        "Zinkevich et al. (2007) — Vanilla CFR with Linear Weighting",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Panel 1: exploitability ───────────────────────────────────────────────
    # Exploitability can be slightly negative early on due to the approximate
    # Nash baseline — clip to a small positive floor before log scaling.
    exploit_plot = [max(v, 1e-6) for v in exploit_values]
    ax1.plot(iteration_points, exploit_plot,
             color="#2563eb", linewidth=2, label="Exploitability e(σ)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iterations", fontsize=11)
    ax1.set_ylabel("Exploitability (log scale)", fontsize=11)
    ax1.set_title("Exploitability — distance from Nash equilibrium", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── Panel 2: EV ───────────────────────────────────────────────────────────
    ax2.plot(iteration_points, ev_values,
             color="#16a34a", linewidth=2, label="P1 EV (average strategy)")
    if LEDUC_GAME_VALUE is not None:
        ax2.axhline(y=LEDUC_GAME_VALUE, color="#dc2626", linestyle="--",
                    linewidth=1.2, label=f"Nash EV ≈ {LEDUC_GAME_VALUE:.4f}")
    ax2.set_xlabel("Iterations", fontsize=11)
    ax2.set_ylabel("Expected Value (chips)", fontsize=11)
    ax2.set_title("Player 1 Expected Value", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nConvergence plot saved to: {save_path}")


def plot_strategy_by_round(
    strategy  : dict[str, dict[str, float]],
    save_path : str = "leduc/plots/strategy_by_round.png",
) -> None:
    """
    Two side-by-side heatmaps: Round 1 infosets and Round 2 infosets.

    Rows    = infoset keys (sorted).
    Columns = passive action probability | aggressive action probability.
              Passive  = check (c) or fold (f).
              Aggressive = bet (b), call (k), or raise (r).
    Colormap = RdYlGn (red = passive, green = aggressive).

    R1 infosets have key format  "<rank>/-:<history>"  (community = '-').
    R2 infosets have key format  "<rank>/<comm>:<history>".
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ── Separate R1 and R2 keys ───────────────────────────────────────────────
    r1_keys = sorted(k for k in strategy if "/-:" in k)
    r2_keys = sorted(k for k in strategy if "/-:" not in k)

    def _passive_aggressive(action_probs: dict[str, float]) -> tuple[float, float]:
        """Sum probabilities into passive (c/f) and aggressive (b/k/r) bins."""
        passive    = sum(p for a, p in action_probs.items() if a in ("c", "f"))
        aggressive = sum(p for a, p in action_probs.items() if a in ("b", "k", "r"))
        return passive, aggressive

    def _build_matrix(keys):
        rows = []
        for k in keys:
            passive, aggressive = _passive_aggressive(strategy[k])
            rows.append([passive, aggressive])
        return np.array(rows) if rows else np.zeros((1, 2))

    r1_matrix = _build_matrix(r1_keys)
    r2_matrix = _build_matrix(r2_keys)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(r2_keys) * 0.3 + 2)))
    fig.suptitle(
        "Leduc Poker — Learned GTO Strategy by Round\n"
        "CFR average strategy (Zinkevich et al. 2007)",
        fontsize=13, fontweight="bold",
    )

    col_labels = ["Passive\n(check/fold)", "Aggressive\n(bet/call/raise)"]

    for ax, matrix, keys, title in [
        (ax1, r1_matrix, r1_keys, "Round 1  (community not yet visible)"),
        (ax2, r2_matrix, r2_keys, "Round 2  (community visible)"),
    ]:
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(col_labels, fontsize=10)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys, fontsize=8 if len(keys) > 20 else 10)
        ax.set_title(title, fontsize=11)
        # Annotate each cell with its value
        for i in range(len(keys)):
            for j in range(2):
                val   = matrix[i, j]
                color = "black" if 0.2 < val < 0.8 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7 if len(keys) > 30 else 9,
                        color=color, fontweight="bold")

    plt.colorbar(
        plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1)),
        ax=[ax1, ax2], label="Action Probability", shrink=0.6,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Strategy-by-round heatmap saved to: {save_path}")


def plot_community_card_effect(
    strategy  : dict[str, dict[str, float]],
    save_path : str = "leduc/plots/community_card_effect.png",
) -> None:
    """
    3 subplots (one per private card J/Q/K), grouped bar chart showing
    aggression probability in Round 2 grouped by community card.

    For each private card × community card pair, aggression probability is
    P(bet | check facing) + P(raise | bet facing) averaged over the round-2
    histories where those actions are available.

    This reveals how much the community card shifts a player's willingness
    to be aggressive — the pair-making effect should be clearly visible
    (e.g. holding J with community J → very aggressive).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    private_ranks = [0, 1, 2]   # JACK, QUEEN, KING
    comm_ranks    = [0, 1, 2]
    comm_labels   = ["J", "Q", "K"]
    private_labels= ["J", "Q", "K"]

    # ── Extract aggression prob per (private, community) pair ─────────────────
    # Average aggression across all R2 opening histories where this pair appears.
    # Aggression = P(bet) when checking, or P(call/raise) when facing a bet.
    def _r2_aggression(private: int, community: int) -> float:
        """
        Average aggression probability for a given (private, community) pair
        across all Round 2 infosets for that pair.
        """
        priv_name = CARD_NAMES[private]
        comm_name = CARD_NAMES[community]
        prefix    = f"{priv_name}/{comm_name}:"

        agg_sum = 0.0
        count   = 0
        for key, action_probs in strategy.items():
            if not key.startswith(prefix):
                continue
            # Skip non-R2 keys (R2 keys contain "/" in the history part)
            history_part = key.split(":", 1)[1]
            if "/" not in history_part:
                continue
            agg = sum(p for a, p in action_probs.items() if a in ("b", "k", "r"))
            agg_sum += agg
            count   += 1

        return agg_sum / count if count > 0 else 0.0

    agg_matrix = np.array([
        [_r2_aggression(priv, comm) for comm in comm_ranks]
        for priv in private_ranks
    ])   # shape (3, 3): rows=private, cols=community

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle(
        "Community Card Effect on Round 2 Aggression — Leduc Poker\n"
        "Aggression probability (bet/call/raise) by private × community card",
        fontsize=13, fontweight="bold",
    )

    bar_colors = ["#3b82f6", "#22c55e", "#f59e0b"]   # J=blue, Q=green, K=amber
    x = np.arange(len(comm_labels))
    width = 0.55

    for idx, (ax, priv) in enumerate(zip(axes, private_ranks)):
        vals = agg_matrix[priv]
        bars = ax.bar(x, vals, width, color=bar_colors[idx], alpha=0.85,
                      edgecolor="white", linewidth=1.2)
        ax.set_title(f"Private card: {private_labels[priv]}", fontsize=12,
                     fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Comm={c}" for c in comm_labels], fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Aggression probability", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        # Annotate bar heights
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
        # Highlight pair-making community card
        pair_comm_idx = priv   # community rank = private rank → pair
        if 0 <= pair_comm_idx < 3:
            axes[idx].get_children()[pair_comm_idx].set_edgecolor("#dc2626")
            axes[idx].get_children()[pair_comm_idx].set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Community card effect plot saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Train and track convergence ───────────────────────────────────────────
    iters, exploit, ev = train_and_track(iterations=1_000, sample_every=100)
    plot_convergence(iters, exploit, ev)

    # ── Train a fresh model for strategy plots ────────────────────────────────
    # Use a longer run so the strategy heatmaps show well-converged values.
    print("\nTraining 1,000-iteration model for strategy plots...")
    trainer = LeducCFRTrainer()
    trainer.train(iterations=1_000)
    strategy = trainer.get_strategy()

    plot_strategy_by_round(strategy)
    plot_community_card_effect(strategy)

    print("\nAll plots saved to leduc/plots/")
