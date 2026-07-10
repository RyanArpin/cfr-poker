"""
holdem/analysis.py
──────────────────
Convergence analysis and strategy plots for the Hold'em MCCFR solver.

Because Hold'em uses Monte Carlo CFR, exact exploitability (full best-
response tree traversal) is expensive.  We use an exploitability estimate:

    For each of N sampled deals, compute:
        p1_br_value = P1's best-response value against the opponent's strategy
        p2_br_value = P2's best-response value against P1's strategy
        sample_exploit = p1_br_value − p2_br_value

    exploit_estimate = mean over N deals

This is a Monte Carlo upper bound on true exploitability but converges
to the true value as N → ∞ and the strategy approaches Nash.

Plots:
    holdem/plots/convergence.png        — EV + exploit estimate vs iterations
    holdem/plots/strategy_by_street.png — four heatmaps (one per street),
                                          rows=bucket, cols=fold/call/raise
    holdem/plots/bucket_distribution.png — bar chart of river bucket frequencies

Reference:
    Lanctot et al. (2009) NeurIPS.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import random

from holdem.holdem_poker import (
    HoldemPoker,
    is_terminal, whose_turn, get_legal_actions,
    street_is_over, advance_street,
    terminal_payoff, infoset_key,
    evaluate_hand, BUCKET_NAMES,
    CHECK, BET, CALL, RAISE, FOLD,
)
from holdem.cfr import HoldemCFRTrainer, HOLDEM_EXPLOITABILITY_TARGET


# ─────────────────────────────────────────────────────────────────────────────
#  Best-response traversal (single sampled deal)
# ─────────────────────────────────────────────────────────────────────────────

def _br_state(
    history   : str,
    buckets   : list[tuple[int, int]],
    street    : int,
    strategy  : dict[str, dict[str, float]],
    br_player : int,
) -> float:
    """
    Best-response value for br_player (pre-computed buckets passed in).
    Returns value from Player 1's perspective.
    """
    if is_terminal(history):
        p1_b, p2_b = buckets[min(street, 3)]
        return terminal_payoff(history, p1_b, p2_b)

    if street_is_over(history) and street < 3:
        return _br_state(
            advance_street(history), buckets, street + 1,
            strategy, br_player,
        )

    player  = whose_turn(history)
    p1_b, p2_b = buckets[street]
    bucket  = p1_b if player == 0 else p2_b
    key     = infoset_key(bucket, history)
    actions = get_legal_actions(history)

    if player == br_player:
        vals = [
            _br_state(history + a, buckets, street, strategy, br_player)
            for a in actions
        ]
        return max(vals) if br_player == 0 else min(vals)
    else:
        probs = strategy.get(key, {a: 1.0 / len(actions) for a in actions})
        return sum(
            probs.get(a, 0.0) * _br_state(
                history + a, buckets, street, strategy, br_player
            )
            for a in actions
        )


def exploitability_estimate(
    game     : HoldemPoker,
    strategy : dict[str, dict[str, float]],
    n_samples: int = 200,
) -> float:
    """
    Estimate exploitability = mean(p1_br − p2_br) over n_samples deals.
    Buckets are pre-computed once per deal for speed.
    """
    total = 0.0
    for _ in range(n_samples):
        deal    = game.deal()
        buckets = [game.get_buckets(deal, s) for s in range(4)]
        p1_br   = _br_state("", buckets, 0, strategy, 0)
        p2_br   = _br_state("", buckets, 0, strategy, 1)
        total  += p1_br - p2_br
    return total / n_samples


# ─────────────────────────────────────────────────────────────────────────────
#  Training with tracking
# ─────────────────────────────────────────────────────────────────────────────

def train_and_track(
    iterations   : int = 10_000,
    sample_every : int = 500,
) -> tuple[list[int], list[float], list[float]]:
    """
    Train MCCFR and record EV + exploitability estimates at intervals.

    Returns
    -------
    (iteration_points, exploit_estimates, ev_estimates)
    """
    game    = HoldemPoker()
    trainer = HoldemCFRTrainer()

    iteration_points = []
    exploit_values   = []
    ev_values        = []

    print(f"Training Hold'em MCCFR for {iterations:,} iterations...")
    print(f"Sampling exploitability every {sample_every} iterations\n")

    completed = 0
    while completed < iterations:
        chunk      = min(sample_every, iterations - completed)
        ev_history = trainer.train(iterations=chunk)
        completed += chunk

        strategy = trainer.get_strategy()

        # Smooth EV over last 100 samples to reduce MC noise
        window  = min(100, len(ev_history))
        ev_est  = sum(ev_history[-window:]) / window

        exploit = exploitability_estimate(game, strategy, n_samples=50)

        iteration_points.append(completed)
        exploit_values.append(exploit)
        ev_values.append(ev_est)

        if completed % (sample_every * 4) == 0 or completed == iterations:
            print(f"  iter {completed:>7,}  |  exploit ≈ {exploit:.4f}"
                  f"  |  EV ≈ {ev_est:.4f}"
                  f"  |  nodes: {len(trainer.nodes):,}")

    print(f"\nFinal exploit estimate: {exploit_values[-1]:.4f}")
    print(f"Final EV estimate:      {ev_values[-1]:.4f}")
    print(f"Infoset nodes:          {len(trainer.nodes):,}")
    print(f"Target:                 ≤ {HOLDEM_EXPLOITABILITY_TARGET}")
    return iteration_points, exploit_values, ev_values


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    iteration_points : list[int],
    exploit_values   : list[float],
    ev_values        : list[float],
    save_path        : str = "holdem/plots/convergence.png",
) -> None:
    """
    Two-panel plot: exploit estimate (log scale) + EV estimate.
    Mirrors kuhn/analysis.py and leduc/analysis.py exactly.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(
        "MCCFR Convergence on Heads-Up Hold'em\n"
        "External Sampling — Lanctot et al. (2009)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    exploit_plot = [max(v, 1e-4) for v in exploit_values if v is not None]
    iters_with_exploit = [iteration_points[i] for i, v in enumerate(exploit_values) if v is not None]
    if exploit_plot:
        ax1.plot(iters_with_exploit, exploit_plot,
                 color="#2563eb", linewidth=2, marker="o", markersize=5,
                 label="Exploit estimate e(σ)")
    ax1.axhline(y=HOLDEM_EXPLOITABILITY_TARGET, color="#dc2626",
                linestyle="--", linewidth=1.2,
                label=f"Target ≤ {HOLDEM_EXPLOITABILITY_TARGET}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iterations", fontsize=11)
    ax1.set_ylabel("Exploitability estimate (log scale)", fontsize=11)
    ax1.set_title("Exploitability estimate — distance from approximate Nash",
                  fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax2.plot(iteration_points, ev_values,
             color="#16a34a", linewidth=2, label="P1 EV estimate (100-iter avg)")
    ax2.axhline(y=0, color="#dc2626", linestyle="--", linewidth=1.2,
                label="EV = 0")
    ax2.set_xlabel("Iterations", fontsize=11)
    ax2.set_ylabel("Expected Value (chips)", fontsize=11)
    ax2.set_title("Player 1 EV Estimate", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nConvergence plot saved to: {save_path}")


def plot_strategy_by_street(
    strategy  : dict[str, dict[str, float]],
    save_path : str = "holdem/plots/strategy_by_street.png",
) -> None:
    """
    Four heatmaps (preflop / flop / turn / river).
    Rows = hand bucket 0–7, columns = fold | call/check | raise/bet.
    Averages action probabilities across all infosets at that (street, bucket).
    Colormap = RdYlGn.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    street_names = ["Preflop", "Flop", "Turn", "River"]
    n_buckets    = 8
    col_labels   = ["Fold", "Call/Check", "Raise/Bet"]

    # Aggregate per (street, bucket)
    # agg[street][bucket] = {col: [probs]}
    agg: list[list[dict[str, list[float]]]] = [
        [{c: [] for c in col_labels} for _ in range(n_buckets)]
        for _ in range(4)
    ]

    for key, action_probs in strategy.items():
        try:
            bucket = int(key.split(":", 1)[0])
            hist   = key.split(":", 1)[1]
            street = hist.count("|")
        except (ValueError, IndexError):
            continue
        if street > 3 or bucket < 0 or bucket > 7:
            continue
        fold_p  = action_probs.get(FOLD,  0.0)
        call_p  = action_probs.get(CALL,  0.0) + action_probs.get(CHECK, 0.0)
        raise_p = action_probs.get(RAISE, 0.0) + action_probs.get(BET,   0.0)
        agg[street][bucket]["Fold"].append(fold_p)
        agg[street][bucket]["Call/Check"].append(call_p)
        agg[street][bucket]["Raise/Bet"].append(raise_p)

    def _mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    matrices = []
    for s in range(4):
        mat = np.zeros((n_buckets, 3))
        for b in range(n_buckets):
            mat[b, 0] = _mean(agg[s][b]["Fold"])
            mat[b, 1] = _mean(agg[s][b]["Call/Check"])
            mat[b, 2] = _mean(agg[s][b]["Raise/Bet"])
        matrices.append(mat)

    fig, axes = plt.subplots(1, 4, figsize=(20, 9), sharey=True)
    fig.suptitle(
        "Hold'em — Learned Strategy by Street and Hand Bucket\n"
        "External Sampling MCCFR (Lanctot et al. 2009)",
        fontsize=13, fontweight="bold",
    )
    bucket_labels = [f"{b}: {BUCKET_NAMES[b]}" for b in range(n_buckets)]

    for ax, matrix, title in zip(axes, matrices, street_names):
        ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(n_buckets))
        ax.set_yticklabels(bucket_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i in range(n_buckets):
            for j in range(3):
                val   = matrix[i, j]
                color = "black" if 0.2 < val < 0.8 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    plt.colorbar(
        plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1)),
        ax=axes.tolist(), label="Action Probability", shrink=0.6,
    )
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Strategy-by-street heatmap saved to: {save_path}")


def plot_bucket_distribution(
    n_samples : int = 50_000,
    save_path : str = "holdem/plots/bucket_distribution.png",
) -> None:
    """
    Bar chart of river hand bucket frequencies across random deals.
    Both players' buckets are pooled.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    game   = HoldemPoker()
    counts = [0] * 8
    for _ in range(n_samples):
        deal = game.deal()
        p1_b, p2_b = game.get_buckets(deal, 3)
        counts[p1_b] += 1
        counts[p2_b] += 1

    total     = sum(counts)
    fractions = [c / total for c in counts]

    bar_colors = ["#94a3b8", "#3b82f6", "#22c55e", "#f59e0b",
                  "#8b5cf6", "#ef4444", "#ec4899", "#0ea5e9"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(8), fractions, color=bar_colors,
                  edgecolor="white", linewidth=1.2, alpha=0.9)
    ax.set_xticks(range(8))
    ax.set_xticklabels(
        [f"{b}\n{BUCKET_NAMES[b]}" for b in range(8)], fontsize=9,
    )
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(
        f"Hand Bucket Distribution at River — Hold'em\n"
        f"({n_samples:,} random deals, both players pooled)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    for bar, frac in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width() / 2, frac + 0.003,
                f"{frac:.1%}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Bucket distribution plot saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    game    = HoldemPoker()
    trainer = HoldemCFRTrainer()

    iterations   = 10_000
    sample_every = 1_000   # only record EV snapshots, no BR calls mid-training

    print(f"Training Hold'em MCCFR for {iterations:,} iterations...")

    iteration_points = []
    ev_values        = []

    start     = time.time()
    completed = 0
    while completed < iterations:
        chunk      = min(sample_every, iterations - completed)
        ev_history = trainer.train(iterations=chunk)
        completed += chunk

        window  = min(100, len(ev_history))
        ev_est  = sum(ev_history[-window:]) / window
        iteration_points.append(completed)
        ev_values.append(ev_est)
        print(f"  iter {completed:>7,}  |  EV ≈ {ev_est:.4f}  |  nodes: {len(trainer.nodes):,}")

    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed:.1f}s")

    # ── Compute exploitability ONCE at the end (50 samples) ──────────────────
    print("Computing final exploitability estimate (50 samples)...")
    strategy     = trainer.get_strategy()
    final_exploit = exploitability_estimate(game, strategy, n_samples=50)
    exploit_values = [None] * (len(iteration_points) - 1) + [final_exploit]

    print(f"Final exploit estimate: {final_exploit:.4f}")
    print(f"Final EV estimate:      {ev_values[-1]:.4f}")
    print(f"Infoset nodes:          {len(trainer.nodes):,}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # For convergence plot, use EV only (exploit is only available at the end)
    plot_convergence(iteration_points, [final_exploit] * len(iteration_points),
                     ev_values)
    plot_strategy_by_street(strategy)
    plot_bucket_distribution()

    print("\nAll plots saved to holdem/plots/")
