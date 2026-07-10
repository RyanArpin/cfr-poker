"""
holdem/holdem_poker.py
──────────────────────
Heads-Up Texas Hold'em game environment for Monte Carlo CFR.

Rules (heads-up, simplified for tractability):
    • Deck: 52 cards.  Rank 2–A (2=0 through A=12), suits s/h/d/c.
    • 2 players.  Small blind (P1) = 1, big blind (P2) = 2.
    • Each player receives 2 hole cards (private).
    • Community cards dealt in stages: flop (3), turn (1), river (1).
    • Betting rounds: preflop, flop, turn, river.
      Bet sizes: preflop/flop = 2, turn/river = 4.  One raise per round.
    • Showdown: standard hand rankings (high card through straight flush).

Card abstraction — why it is necessary:
    The full Hold'em game tree has ~10^14 information sets, making exact
    CFR completely intractable.  We apply *card abstraction*: map each
    private hand (2 hole cards + visible community cards) to one of 8
    hand-strength buckets.  This reduces the infoset space to a size
    MC-CFR can explore in reasonable time.

    Hand buckets (0 = weakest, 7 = strongest):
        0 — high card
        1 — one pair
        2 — two pair
        3 — three of a kind
        4 — straight
        5 — flush
        6 — full house
        7 — four of a kind or straight flush

    Infoset key: "<bucket>:<history>"

History format:
    A single flat string of action characters, with "|" separating streets.
    Examples:
        ""              — preflop, no actions yet
        "kr"            — preflop: P1 called (k), P2 raised (r)
        "kc|"           — flop started (preflop closed: P1 called, P2 checked)
        "kc|cb"         — flop: P1 checked (c), P2 bet (b)
        "kc|cc|"        — turn started

    The number of "|" in the history tells you which street we are on:
        0 pipes → street 0 (preflop)
        1 pipe  → street 1 (flop)
        2 pipes → street 2 (turn)
        3 pipes → street 3 (river)

    A history ends at a terminal state: fold anywhere, or river betting done.

Action characters (consistent with Leduc):
    c = check  (or big-blind option preflop)
    k = call   (avoids collision with c=check)
    b = bet
    r = raise
    f = fold
"""

from __future__ import annotations
import random
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
#  Card representation
#
#  card integer 0–51: rank = card // 4  (0=2 … 12=Ace), suit = card % 4
# ─────────────────────────────────────────────────────────────────────────────

DECK_SIZE  = 52
NUM_RANKS  = 13
NUM_SUITS  = 4

RANK_NAMES = {i: r for i, r in enumerate("23456789TJQKA")}
SUIT_NAMES = {0: "s", 1: "h", 2: "d", 3: "c"}


def card_rank(card: int) -> int:
    return card // 4


def card_suit(card: int) -> int:
    return card % 4


def card_name(card: int) -> str:
    return RANK_NAMES[card_rank(card)] + SUIT_NAMES[card_suit(card)]


# ─────────────────────────────────────────────────────────────────────────────
#  Action constants
# ─────────────────────────────────────────────────────────────────────────────

CHECK = "c"
BET   = "b"
CALL  = "k"    # 'k' avoids collision with CHECK='c'
RAISE = "r"
FOLD  = "f"

# Bet size per street index (0=preflop, 1=flop, 2=turn, 3=river)
BET_SIZE = {0: 2, 1: 2, 2: 4, 3: 4}

SMALL_BLIND = 1
BIG_BLIND   = 2

# ─────────────────────────────────────────────────────────────────────────────
#  Street-closed detection
#
#  Identical closed-action sequences to Leduc's round detection.
#  A street ends (without fold) when the action sequence matches one of:
#    "cc"    check-check
#    "kc"    call then check (big-blind option or similar)
#    "bk"    bet-call
#    "brk"   bet-raise-call
#    "cbk"   check-bet-call
#    "cbrk"  check-bet-raise-call
#
#  A street ends WITH a fold when the sequence matches:
#    "f"     first actor folds immediately
#    "bf"    bet-fold
#    "brf"   bet-raise-fold
#    "cbf"   check-bet-fold
#    "cbrf"  check-bet-raise-fold
#    "krf"   call-raise-fold  (preflop: SB calls, BB raises, SB folds)
# ─────────────────────────────────────────────────────────────────────────────

_CLOSED_NO_FOLD   = {"cc", "kc", "bk", "brk", "cbk", "cbrk"}
_CLOSED_WITH_FOLD = {"f", "bf", "brf", "cbf", "cbrf", "krf"}
_CLOSED           = _CLOSED_NO_FOLD | _CLOSED_WITH_FOLD


def _street_actions(history: str) -> str:
    """Return the action sequence for the current (last) street."""
    return history.split("|")[-1]


def _current_street(history: str) -> int:
    """0=preflop, 1=flop, 2=turn, 3=river."""
    return history.count("|")


# ─────────────────────────────────────────────────────────────────────────────
#  Public game-tree functions
# ─────────────────────────────────────────────────────────────────────────────

def is_terminal(history: str) -> bool:
    """
    Return True if the hand is completely over.

    Terminal when:
      • Any street segment contains a fold sequence, OR
      • The river segment is a closed-no-fold sequence (showdown).
    """
    for seg in history.split("|"):
        if seg in _CLOSED_WITH_FOLD:
            return True
    parts = history.split("|")
    if len(parts) == 4 and parts[3] in _CLOSED_NO_FOLD:
        return True
    return False


def whose_turn(history: str) -> int:
    """0=P1, 1=P2.  P1 acts first on every street; players alternate."""
    return len(_street_actions(history)) % 2


def get_legal_actions(history: str) -> list[str]:
    """Legal actions at a non-terminal decision node."""
    if is_terminal(history):
        raise ValueError(f"No actions at terminal history '{history}'")
    sa = _street_actions(history)
    if sa and sa[-1] in (BET, RAISE):
        actions = [FOLD, CALL]
        if sa.count(RAISE) < 1:
            actions.append(RAISE)
        return actions
    return [CHECK, BET]


def street_is_over(history: str) -> bool:
    """True when the current street's action sequence is complete."""
    return _street_actions(history) in _CLOSED


def advance_street(history: str) -> str:
    """Append '|' to start the next betting street."""
    return history + "|"


# ─────────────────────────────────────────────────────────────────────────────
#  Payoff computation
# ─────────────────────────────────────────────────────────────────────────────

def _chips_contributed(history: str) -> tuple[int, int]:
    """Return (p1_chips, p2_chips) total including blinds."""
    p1 = SMALL_BLIND
    p2 = BIG_BLIND
    for street_idx, seg in enumerate(history.split("|")):
        bet = BET_SIZE[min(street_idx, 3)]
        for action_idx, action in enumerate(seg):
            if action in (BET, CALL, RAISE):
                if action_idx % 2 == 0:
                    p1 += bet
                else:
                    p2 += bet
    return p1, p2


def terminal_payoff(history: str, p1_bucket: int, p2_bucket: int) -> float:
    """
    Player 1's payoff at a terminal history.

    Fold payoffs are card-independent.  Showdown uses bucket comparison:
    higher bucket wins; equal buckets split (return 0).
    """
    assert is_terminal(history), f"'{history}' is not terminal"

    p1_chips, p2_chips = _chips_contributed(history)

    # ── Find which segment contains a fold ────────────────────────────────────
    for seg in history.split("|"):
        if seg in _CLOSED_WITH_FOLD:
            folder = (len(seg) - 1) % 2
            return (-p1_chips) if folder == 0 else p2_chips

    # ── Showdown ──────────────────────────────────────────────────────────────
    if p1_bucket > p2_bucket:
        return p2_chips
    if p2_bucket > p1_bucket:
        return -p1_chips
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Hand evaluation and card abstraction
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_hand(hole_cards: list[int], community_cards: list[int]) -> int:
    """
    Best 5-card hand bucket (0–7) from hole + community cards.

    Bucket mapping:
        0 high card, 1 one pair, 2 two pair, 3 three of a kind,
        4 straight, 5 flush, 6 full house, 7 four of a kind / straight flush.
    """
    all_cards = hole_cards + community_cards
    if len(all_cards) < 2:
        return 0
    if len(all_cards) <= 5:
        return _classify(all_cards)
    return max(_classify(list(c)) for c in combinations(all_cards, 5))


def _classify(cards: list[int]) -> int:
    """Classify a 2–5 card hand into a bucket."""
    ranks  = sorted([card_rank(c) for c in cards], reverse=True)
    suits  = [card_suit(c) for c in cards]

    rank_counts: dict[int, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    counts = sorted(rank_counts.values(), reverse=True)

    n           = len(cards)
    is_flush    = (n == 5) and (len(set(suits)) == 1)
    is_straight = False
    if n == 5 and len(rank_counts) == 5:
        if max(ranks) - min(ranks) == 4:
            is_straight = True
        elif set(ranks) == {12, 0, 1, 2, 3}:   # A-2-3-4-5 wheel
            is_straight = True

    if is_straight and is_flush:
        return 7
    if counts[0] == 4:
        return 7
    if counts[0] == 3 and len(counts) > 1 and counts[1] == 2:
        return 6
    if is_flush:
        return 5
    if is_straight:
        return 4
    if counts[0] == 3:
        return 3
    if counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        return 2
    if counts[0] == 2:
        return 1
    return 0


BUCKET_NAMES = {
    0: "High card",
    1: "One pair",
    2: "Two pair",
    3: "Three of a kind",
    4: "Straight",
    5: "Flush",
    6: "Full house",
    7: "Four of a kind / Straight flush",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Infoset key
# ─────────────────────────────────────────────────────────────────────────────

def infoset_key(bucket: int, history: str) -> str:
    """
    "<bucket>:<history>"  — e.g. "3:" (preflop, bucket 3), "1:kc|cb" (flop).
    """
    return f"{bucket}:{history}"


# ─────────────────────────────────────────────────────────────────────────────
#  HoldemPoker class
# ─────────────────────────────────────────────────────────────────────────────

class HoldemPoker:
    """
    Heads-Up Texas Hold'em game environment.

    The CFR trainer never sees raw cards — only bucket values.
    """

    def is_terminal(self, history: str) -> bool:
        return is_terminal(history)

    def whose_turn(self, history: str) -> int:
        return whose_turn(history)

    def get_legal_actions(self, history: str) -> list[str]:
        return get_legal_actions(history)

    def get_payoff(self, history: str, p1_bucket: int, p2_bucket: int) -> float:
        return terminal_payoff(history, p1_bucket, p2_bucket)

    def get_infoset_key(self, bucket: int, history: str) -> str:
        return infoset_key(bucket, history)

    def evaluate_hand(self, hole_cards: list[int], community_cards: list[int]) -> int:
        return evaluate_hand(hole_cards, community_cards)

    def deal(self) -> dict:
        """
        Sample a complete random hand.

        Returns dict with p1_hole, p2_hole, flop, turn, river.
        """
        deck = list(range(DECK_SIZE))
        random.shuffle(deck)
        return {
            "p1_hole" : deck[0:2],
            "p2_hole" : deck[2:4],
            "flop"    : deck[4:7],
            "turn"    : deck[7:8],
            "river"   : deck[8:9],
        }

    def get_buckets(self, deal: dict, street: int) -> tuple[int, int]:
        """
        (p1_bucket, p2_bucket) for the given street.
        street: 0=preflop (no community), 1=flop, 2=turn, 3=river.
        """
        if street == 0:
            comm = []
        elif street == 1:
            comm = deal["flop"]
        elif street == 2:
            comm = deal["flop"] + deal["turn"]
        else:
            comm = deal["flop"] + deal["turn"] + deal["river"]
        return (
            evaluate_hand(deal["p1_hole"], comm),
            evaluate_hand(deal["p2_hole"], comm),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone sanity checks
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import collections

    game = HoldemPoker()

    print("=" * 60)
    print("Hold'em Environment — Sanity Checks")
    print("=" * 60)

    # ── 1. Terminal detection ─────────────────────────────────────────────────
    print("\n[1] Terminal detection:")
    cases = [
        ("f",           True),
        ("krf",         True),
        ("kc|f",        True),
        ("kc|cc|cc|cc", True),
        ("",            False),
        ("k",           False),
        ("kc|",         False),
        ("kc|cc|cc|c",  False),
    ]
    ok = True
    for h, exp in cases:
        got = game.is_terminal(h)
        s = "✓" if got == exp else "✗"
        print(f"    {s} is_terminal('{h}') = {got}  (expected {exp})")
        if got != exp:
            ok = False
    if ok:
        print("    All terminal checks passed ✓")

    # ── 2. Whose turn ─────────────────────────────────────────────────────────
    print("\n[2] Whose turn:")
    assert game.whose_turn("")     == 0
    assert game.whose_turn("k")    == 1
    assert game.whose_turn("kr")   == 0
    assert game.whose_turn("kc|")  == 0
    assert game.whose_turn("kc|c") == 1
    print("    All whose_turn checks passed ✓")

    # ── 3. Legal actions ──────────────────────────────────────────────────────
    print("\n[3] Legal actions:")
    assert set(game.get_legal_actions(""))    == {"c", "b"}
    assert set(game.get_legal_actions("b"))   == {"f", "k", "r"}
    assert set(game.get_legal_actions("br"))  == {"f", "k"}
    assert set(game.get_legal_actions("kc|")) == {"c", "b"}
    print("    All legal action checks passed ✓")

    # ── 4. Hand evaluation ────────────────────────────────────────────────────
    print("\n[4] Hand evaluation:")
    # Pair of Aces: A♠=48, A♥=49, board has no straights/flushes
    assert evaluate_hand([48, 49], [0, 13, 30, 39, 4]) == 1, "One pair"
    assert evaluate_hand([0, 5],   [])                  == 0, "High card"
    assert evaluate_hand([48, 49], [50, 51, 4])         == 7, "Four aces"
    assert evaluate_hand([44, 40], [45, 41, 0])         == 2, "Two pair"
    print("    All hand evaluation checks passed ✓")

    # ── 5. Payoffs ────────────────────────────────────────────────────────────
    print("\n[5] Payoffs:")
    assert terminal_payoff("f",           0, 7) == -1,  "P1 folds preflop"
    assert terminal_payoff("kc|cc|cc|cc", 3, 1)  > 0,   "P1 wins showdown"
    assert terminal_payoff("kc|cc|cc|cc", 2, 2) == 0.0, "Split pot"
    print("    All payoff checks passed ✓")

    # ── 6. Bucket distribution ────────────────────────────────────────────────
    print("\n[6] River bucket distribution (5,000 deals):")
    counts: dict[int, int] = collections.Counter()
    for _ in range(5_000):
        d = game.deal()
        p1b, p2b = game.get_buckets(d, 3)
        counts[p1b] += 1
        counts[p2b] += 1
    total = sum(counts.values())
    for b in range(8):
        print(f"    Bucket {b} {BUCKET_NAMES[b]:<38}: {counts[b]/total:.1%}")

    print("\n" + "=" * 60)
    print("Phase 5 environment verified. Ready for HoldemCFRTrainer.")
    print("=" * 60)
