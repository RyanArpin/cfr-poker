"""
kuhn_poker.py
─────────────
Kuhn Poker game environment for CFR.

Kuhn Poker rules:
    • Deck: J (Jack=0), Q (Queen=1), K (King=2)
    • 2 players, each receives 1 card. 1 card is burned (not used).
    • Both players ante 1 chip before play.
    • 1 betting round: check or bet (1 chip). If bet, opponent can fold or call.
    • Showdown: higher card wins the pot.

Node identifier string format (used throughout the project):
    "<P1 card><P2 card>:<action history>"
    e.g.  "KQ:cb"  means P1 holds King, P2 holds Queen, history is check-bet.
    The action history uses:
        'c' = check
        'b' = bet
        'f' = fold (only legal in response to a bet)

This file defines:
    KuhnPoker   — the game environment class
    GameNode    — a lightweight representation of one node in the game tree
"""

from itertools import permutations


# ─────────────────────────────────────────────────────────────────────────────
#  Card constants
#  We use integers internally; the mapping is J=0, Q=1, K=2.
#  Higher integer = stronger card at showdown.
# ─────────────────────────────────────────────────────────────────────────────
JACK  = 0
QUEEN = 1
KING  = 2

CARD_NAMES = {JACK: "J", QUEEN: "Q", KING: "K"}
CARD_FROM_NAME = {"J": JACK, "Q": QUEEN, "K": KING}


# ─────────────────────────────────────────────────────────────────────────────
#  Action constants
#  We keep actions as single characters so the history string stays readable.
# ─────────────────────────────────────────────────────────────────────────────
CHECK = "c"
BET   = "b"
CALL  = "c"   # NOTE: In a bet-response context 'c' means *call*, not check.
              #       Context (length of history) disambiguates.
FOLD  = "f"

# The two possible actions at every decision point in Kuhn Poker.
# Player 1 can: check ("c") or bet ("b") from an empty history.
# Player 2 after a check can: check ("c") or bet ("b").
# Player 2 after a bet can: fold ("f") or call ("c").
# Player 1 after check-bet can: fold ("f") or call ("c").
ACTIONS = [CHECK, BET]          # used when no bet is facing the player
RESPONSE_ACTIONS = [FOLD, CALL] # used when a bet is facing the player


# ─────────────────────────────────────────────────────────────────────────────
#  Terminal history detection
#
#  A history is terminal when play has ended (no more decisions needed).
#  In Kuhn Poker the terminal histories are exactly:
#      "cc"   — both players checked  → showdown
#      "bc"   — bet then call         → showdown
#      "bf"   — bet then fold         → P1 wins unopposed
#      "cbc"  — check, bet, call      → showdown
#      "cbf"  — check, bet, fold      → P2 wins unopposed
# ─────────────────────────────────────────────────────────────────────────────
TERMINAL_HISTORIES = {"cc", "bc", "bf", "cbc", "cbf"}


def is_terminal(history: str) -> bool:
    """Return True if this action history is a terminal node (game over)."""
    return history in TERMINAL_HISTORIES


def whose_turn(history: str) -> int:
    """
    Return which player acts next (0 = Player 1, 1 = Player 2).

    Kuhn Poker turn order:
        ""    → P1 acts   (start of hand)
        "c"   → P2 acts   (P1 checked)
        "b"   → P2 acts   (P1 bet)
        "cb"  → P1 acts   (P1 checked, P2 bet)

    History length mod 2 doesn't work cleanly here because we need to
    know the actual history, not just its length.  We handle each case.
    """
    if history in ("", ):
        return 0    # Player 1 acts first
    if history in ("c", "b"):
        return 1    # Player 2 responds to P1's first action
    if history in ("cb",):
        return 0    # Player 1 responds to P2's bet after checking
    # Should never reach here for non-terminal histories
    raise ValueError(f"Cannot determine whose turn for history: '{history}'")


def get_legal_actions(history: str) -> list[str]:
    """
    Return the list of legal actions at this (non-terminal) history.

    If a bet is currently facing the acting player, they must fold or call.
    Otherwise, they may check or bet.

    A bet is "facing" the player when the last character of the history is 'b'.
    """
    if is_terminal(history):
        raise ValueError(f"No actions at terminal history '{history}'")

    # Is the last action a bet? If so, current player must respond.
    if history and history[-1] == BET:
        return [FOLD, CALL]   # response to a bet: fold or call
    else:
        return [CHECK, BET]   # no bet facing: check or bet


# ─────────────────────────────────────────────────────────────────────────────
#  Terminal payoff function
#
#  From Zinkevich et al. (2007): the utility u_i(z) of a terminal history z
#  for player i is the number of chips won by player i.
#
#  Both players ante 1 chip → pot starts at 2.
#  A bet adds 1 chip for a total pot of 4 if called.
#  Returns the payoff for *Player 1* (P2's payoff is the negative of P1's
#  because Kuhn Poker is zero-sum).
# ─────────────────────────────────────────────────────────────────────────────
def terminal_payoff(history: str, p1_card: int, p2_card: int) -> float:
    """
    Compute Player 1's payoff at a terminal history.

    Parameters
    ----------
    history : str
        The terminal action history (must be in TERMINAL_HISTORIES).
    p1_card : int
        Card held by Player 1 (JACK=0, QUEEN=1, KING=2).
    p2_card : int
        Card held by Player 2 (JACK=0, QUEEN=1, KING=2).

    Returns
    -------
    float
        Chips gained (positive) or lost (negative) by Player 1.

    Payoff table (from Player 1's perspective):
    ┌─────────┬─────────────────────────────────────────────┐
    │ History │ Outcome                                     │
    ├─────────┼─────────────────────────────────────────────┤
    │ "cc"    │ showdown, pot=2 → winner gets +1, loser -1 │
    │ "bc"    │ showdown, pot=4 → winner gets +2, loser -2 │ (called a bet)
    │ "bf"    │ P2 folded      → P1 wins +1 regardless     │
    │ "cbc"   │ showdown, pot=4 → winner gets +2, loser -2 │
    │ "cbf"   │ P1 folded      → P1 loses -1               │
    └─────────┴─────────────────────────────────────────────┘
    """
    assert history in TERMINAL_HISTORIES, f"'{history}' is not terminal"

    # ── Fold outcomes (no showdown needed) ──────────────────────────────────
    if history == "bf":
        # P1 bet, P2 folded → P1 wins the ante (1 chip)
        return 1

    if history == "cbf":
        # P1 checked, P2 bet, P1 folded → P1 loses the ante (1 chip)
        return -1

    # ── Showdown outcomes ────────────────────────────────────────────────────
    # Determine who wins based on card rank (higher int = better card)
    p1_wins_showdown = p1_card > p2_card

    if history == "cc":
        # Both checked → pot is 2 chips (1 ante each)
        # Winner gains 1 chip (opponent's ante); loser net -1
        return 1 if p1_wins_showdown else -1

    if history in ("bc", "cbc"):
        # There was a bet and a call → pot is 4 chips (1 ante + 1 bet each)
        # Winner gains 2 chips; loser net -2
        return 2 if p1_wins_showdown else -2

    # Should never reach here if history is validated
    raise ValueError(f"Unhandled terminal history: '{history}'")


# ─────────────────────────────────────────────────────────────────────────────
#  Information Set (Infoset) key
#
#  In Zinkevich et al., an *information set* groups together all game states
#  that a player cannot distinguish given their private information.
#
#  In Kuhn Poker, Player i knows only:
#    - their own card
#    - the public action history
#  They do NOT know the opponent's card.
#
#  So the infoset key for player i is: "<own card letter>:<history>"
#  e.g.  "K:cb"  — Player 1 holds King, history is check-bet
#        "J:b"   — Player 2 holds Jack, Player 1 bet
# ─────────────────────────────────────────────────────────────────────────────
def infoset_key(card: int, history: str) -> str:
    """
    Build the information set key used to index into CFR nodes.

    Parameters
    ----------
    card    : int  — the acting player's private card (JACK/QUEEN/KING)
    history : str  — the public action history so far

    Returns
    -------
    str  — e.g. "K:cb", "J:b", "Q:"
    """
    return f"{CARD_NAMES[card]}:{history}"


# ─────────────────────────────────────────────────────────────────────────────
#  KuhnPoker class
#  Encapsulates the game environment and all deal iteration logic.
# ─────────────────────────────────────────────────────────────────────────────
class KuhnPoker:
    """
    Kuhn Poker game environment.

    This class doesn't run CFR itself — it provides:
      • The list of all possible card deals
      • Methods to step through the game tree from any state
      • Payoff computation
      • Infoset key generation

    The CFR algorithm (Phase 2) will import and use this class.
    """

    CARDS = [JACK, QUEEN, KING]   # all cards in the deck

    def __init__(self):
        # All ordered 2-card deals from {J, Q, K}.
        # Each deal is (p1_card, p2_card).
        # There are 3 × 2 = 6 possible deals.
        self.deals = list(permutations(self.CARDS, 2))

    # ── Public interface ──────────────────────────────────────────────────────

    def is_terminal(self, history: str) -> bool:
        return is_terminal(history)

    def whose_turn(self, history: str) -> int:
        return whose_turn(history)

    def get_legal_actions(self, history: str) -> list[str]:
        return get_legal_actions(history)

    def get_payoff(self, history: str, p1_card: int, p2_card: int) -> float:
        """Player 1's payoff at a terminal node (P2's is the negation)."""
        return terminal_payoff(history, p1_card, p2_card)

    def get_infoset_key(self, card: int, history: str) -> str:
        return infoset_key(card, history)

    def get_all_deals(self) -> list[tuple[int, int]]:
        """
        Return all 6 possible (p1_card, p2_card) deals.
        CFR will iterate over these to compute expected values.
        """
        return self.deals

    # ── Tree enumeration helpers ──────────────────────────────────────────────

    def enumerate_game_tree(self) -> dict:
        """
        Walk the full game tree and return a dict mapping each
        (p1_card, p2_card, history) → node_type.

        node_type is one of: 'terminal', 'p1_decision', 'p2_decision'

        This is mostly for visualization and debugging — CFR doesn't need it.
        """
        tree = {}
        all_histories = self._all_histories()

        for p1_card, p2_card in self.deals:
            for history in all_histories:
                if is_terminal(history):
                    tree[(p1_card, p2_card, history)] = "terminal"
                else:
                    actor = whose_turn(history)
                    node_type = "p1_decision" if actor == 0 else "p2_decision"
                    tree[(p1_card, p2_card, history)] = node_type

        return tree

    def _all_histories(self) -> list[str]:
        """Return all possible action histories (terminal and non-terminal)."""
        return [
            "",       # start
            "c",      # P1 checked
            "b",      # P1 bet
            "cc",     # both checked     → terminal
            "cb",     # P1 check, P2 bet
            "bc",     # P1 bet, P2 called → terminal
            "bf",     # P1 bet, P2 folded → terminal
            "cbc",    # P1 c, P2 b, P1 called → terminal
            "cbf",    # P1 c, P2 b, P1 folded → terminal
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test — run this file directly to verify the environment
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    game = KuhnPoker()

    print("=" * 60)
    print("Kuhn Poker Environment — Sanity Checks")
    print("=" * 60)

    # ── Check 1: all 6 deals ─────────────────────────────────────────────────
    print("\n[1] All possible deals (p1_card, p2_card):")
    for p1, p2 in game.get_all_deals():
        print(f"    P1={CARD_NAMES[p1]}  P2={CARD_NAMES[p2]}")

    # ── Check 2: legal actions at every non-terminal history ─────────────────
    print("\n[2] Legal actions at each non-terminal history:")
    non_terminals = ["", "c", "b", "cb"]
    for h in non_terminals:
        actions = game.get_legal_actions(h)
        player  = game.whose_turn(h)
        print(f"    history='{h}'  player={player+1}  actions={actions}")

    # ── Check 3: payoffs at every terminal history for all deals ─────────────
    print("\n[3] Terminal payoffs (Player 1's perspective):")
    terminals = ["cc", "bc", "bf", "cbc", "cbf"]
    for hist in terminals:
        print(f"\n  History = '{hist}'")
        for p1, p2 in game.get_all_deals():
            payoff = game.get_payoff(hist, p1, p2)
            print(f"    P1={CARD_NAMES[p1]} P2={CARD_NAMES[p2]} → payoff={payoff:+.0f}")

    # ── Check 4: infoset keys ─────────────────────────────────────────────────
    print("\n[4] Infoset key examples:")
    examples = [(KING, ""), (QUEEN, "b"), (JACK, "cb")]
    for card, hist in examples:
        key = game.get_infoset_key(card, hist)
        print(f"    card={CARD_NAMES[card]}  history='{hist}'  → key='{key}'")

    # ── Check 5: game tree structure ──────────────────────────────────────────
    print("\n[5] Unique non-terminal histories in the game tree:")
    tree = game.enumerate_game_tree()
    for hist in game._all_histories():
        if not is_terminal(hist):
            node_type = tree[(JACK, QUEEN, hist)]  # pick any deal, type is same
            actions   = game.get_legal_actions(hist)
            print(f"    '{hist}'  ({node_type})  → {actions}")

    # ── Check 6: verify known payoffs from the Zinkevich paper example ────────
    print("\n[6] Verifying known payoffs from Zinkevich et al.:")
    # K vs J: showdown, K wins. Both checked → P1 wins +1
    assert game.get_payoff("cc", KING, JACK) == 1
    # J vs K: showdown, K wins. P1 bet, P2 called → P1 loses -2
    assert game.get_payoff("bc", JACK, KING) == -2
    # Any card vs any card: P2 folded → P1 wins +1
    assert game.get_payoff("bf", JACK, KING) == 1
    # Any card vs any card: P1 folded → P1 loses -1
    assert game.get_payoff("cbf", KING, JACK) == -1
    print("    All assertions passed ✓")

    print("\n" + "=" * 60)
    print("Phase 1 complete. Game environment verified.")
    print("=" * 60)