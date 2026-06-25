"""
leduc/leduc_poker.py
────────────────────
Leduc Poker game environment for CFR.

Leduc Poker rules:
    • Deck: J, J, Q, Q, K, K  (6 cards, two of each rank)
    • 2 players, each receives 1 private card face-down.
    • Round 1: both ante 1 chip. Betting with bet size = 2. Max 1 raise.
    • Community card dealt face-up between rounds (chance node).
    • Round 2: betting with bet size = 4. Max 1 raise.
    • Showdown: a *pair* (private card matches community) beats any non-pair.
      Among non-pairs, higher rank wins.

Reference:
    Southey et al. (2005) "Bayes' Bluff: Opponent Modelling in Poker"
    — standard Leduc Poker specification.

History string format:
    "<round1_actions>/<community_card>/<round2_actions>"
    e.g. "bk/Q/br" — R1 bet-called, community Queen, R2 bet-raise

    Before the community card has been revealed the history is just the
    round-1 action sequence, e.g. "b", "bk".

    After the community card arrives the separator "/" is inserted, then
    round-2 actions follow the second "/".

Action characters (chosen to avoid ambiguity):
    c = check
    b = bet
    k = call  ('c' already means check; 'k' for "call" avoids the clash)
    r = raise
    f = fold

Infoset key format:
    "<private_rank>/<community_or_->:<history>"

    Before community:  "K/-:b"      — holds King, round 1, facing a bet
    After  community:  "K/J:bk/J/c" — holds King, community Jack, round 2

Dealing model:
    The 6-card deck (J,J,Q,Q,K,K) is dealt without replacement.  There are
    6×5 = 30 ordered deals for the two private cards.  get_all_private_deals()
    returns all 30 (p1_card_index, p2_card_index) pairs drawn from the indexed
    deck so that CFR weights each deal uniformly (1/30).  The rank of each
    card is deck[index], so same-rank pairs naturally appear twice (JJ, QQ, KK
    each appear twice: indices (0,1) and (1,0), etc.) giving them 4/30 weight
    compared to 2/30 for mixed-rank pairs — matching true deal probabilities.

    get_community_cards() then conditions on which two physical cards were
    removed, returning the correct posterior over the remaining 4 cards.

Why the design matches KuhnPoker's interface:
    LeducCFRTrainer.train() iterates over self.game.get_all_private_deals(),
    calls self.game.is_terminal(), .whose_turn(), .get_legal_actions(),
    .get_payoff(), and .get_infoset_key() — all are reproduced here.

    The only extension is the chance node between rounds:
        is_chance_node(history) → True when R1 ended without a fold
        get_community_cards(p1_idx, p2_idx) → [(rank, prob), ...]
    LeducCFRTrainer uses these to enumerate community cards.
"""

from __future__ import annotations
from itertools import permutations


# ─────────────────────────────────────────────────────────────────────────────
#  Card constants
# ─────────────────────────────────────────────────────────────────────────────
JACK  = 0
QUEEN = 1
KING  = 2

CARD_NAMES    = {JACK: "J", QUEEN: "Q", KING: "K"}
CARD_FROM_NAME = {"J": JACK, "Q": QUEEN, "K": KING}

# Full Leduc deck: two of each rank, indexed 0–5.
# Index encodes the physical card; rank = DECK[index].
# This lets us enumerate all 30 ordered deals while keeping payoff logic
# purely rank-based (DECK[i] gives the rank for card index i).
DECK = [JACK, JACK, QUEEN, QUEEN, KING, KING]


# ─────────────────────────────────────────────────────────────────────────────
#  Action constants
# ─────────────────────────────────────────────────────────────────────────────
CHECK = "c"
BET   = "b"
CALL  = "k"   # 'k' for call, to avoid clash with CHECK='c'
RAISE = "r"
FOLD  = "f"

# Leduc bet sizes per round
BET_SIZE = {1: 2, 2: 4}   # round → chip amount


# ─────────────────────────────────────────────────────────────────────────────
#  History parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_history(history: str) -> tuple[str, str | None, str | None]:
    """
    Parse a history string into its components.

    Returns
    -------
    (r1_actions, community, r2_actions)

    Examples
    --------
    ""          → ("",   None,  None)
    "b"         → ("b",  None,  None)
    "bk/Q"      → ("bk", "Q",   None)   ← chance node just occurred
    "bk/Q/"     → ("bk", "Q",   "")     ← round 2 started, no actions yet
    "bk/Q/br"   → ("bk", "Q",   "br")
    """
    parts = history.split("/")
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        # e.g. "bk/Q" — community card was just appended, round 2 not started
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"Malformed history: '{history}'")


def _current_round(history: str) -> int:
    """Return 1 or 2 — the betting round currently in progress."""
    _, community, r2 = _split_history(history)
    if community is None:
        return 1
    return 2


def _round_actions(history: str) -> str:
    """Return only the action sequence for the current round."""
    r1, community, r2 = _split_history(history)
    if community is None:
        return r1       # still in round 1
    if r2 is None:
        return ""       # round 2 just started, no actions yet
    return r2


# ─────────────────────────────────────────────────────────────────────────────
#  Round-1 terminal detection
#
#  Round 1 ends when:
#    a) Someone folds → whole game is terminal
#    b) Both players have acted and neither raised, OR a raise was called/folded
#
#  Possible round-1-over sequences (without a fold):
#    "cc"    check-check
#    "bk"    bet-call
#    "br"    wait… no: br = bet-raise, still in R1 (P1 must respond)
#    "brk"   bet-raise-call
#    "brf"   bet-raise-fold  ← fold → game terminal
#    "cbk"   check-bet-call
#    "cbf"   check-bet-fold  ← game terminal
#    "cbrk"  check-bet-raise-call
#    "cbrf"  check-bet-raise-fold ← game terminal
# ─────────────────────────────────────────────────────────────────────────────

# Round-1 sequences where play ended WITHOUT a fold (→ chance node next)
_R1_NONFOLD_TERMINALS = {"cc", "bk", "brk", "cbk", "cbrk"}

# Round-1 sequences where play ended WITH a fold (→ whole game terminal)
_R1_FOLD_TERMINALS = {"bf", "brf", "cbf", "cbrf"}


def _r1_ended_no_fold(r1_actions: str) -> bool:
    return r1_actions in _R1_NONFOLD_TERMINALS


def _r1_ended_with_fold(r1_actions: str) -> bool:
    return r1_actions in _R1_FOLD_TERMINALS


# ─────────────────────────────────────────────────────────────────────────────
#  Round-2 terminal detection
#
#  Same structure as round 1 but the action set is identical.
#    "cc"   check-check  → showdown
#    "bk"   bet-call     → showdown
#    "brk"  bet-raise-call → showdown
#    "cbk"  check-bet-call → showdown
#    "cbrk" check-bet-raise-call → showdown
#    "bf"/"brf"/"cbf"/"cbrf" → fold (game terminal)
# ─────────────────────────────────────────────────────────────────────────────

_R2_NONFOLD_TERMINALS = {"cc", "bk", "brk", "cbk", "cbrk"}
_R2_FOLD_TERMINALS    = {"bf", "brf", "cbf", "cbrf"}

def _r2_ended(r2_actions: str) -> bool:
    return r2_actions in _R2_NONFOLD_TERMINALS or r2_actions in _R2_FOLD_TERMINALS


# ─────────────────────────────────────────────────────────────────────────────
#  Bet-facing detection
#
#  A bet is currently "facing" the acting player when the last action in
#  the current round is 'b' (bet) or 'r' (raise). In both cases the player
#  can fold, call, or (if max raises not hit) raise.
# ─────────────────────────────────────────────────────────────────────────────

def _bet_is_facing(round_actions: str) -> bool:
    return bool(round_actions) and round_actions[-1] in (BET, RAISE)


def _raise_count(round_actions: str) -> int:
    """Count raises in the current round (max 1 allowed per round)."""
    return round_actions.count(RAISE)


# ─────────────────────────────────────────────────────────────────────────────
#  Public game-tree functions
# ─────────────────────────────────────────────────────────────────────────────

def is_terminal(history: str) -> bool:
    """
    Return True if the game is fully over (no more player decisions).

    Note: a history like "bk/Q" is NOT terminal — it is a chance node
    (community card was just placed). is_terminal() returns False there.
    The chance node is handled by is_chance_node().
    """
    r1, community, r2 = _split_history(history)

    # Round 1 ended with a fold — game over
    if _r1_ended_with_fold(r1):
        return True

    # Community card not yet dealt — not terminal (still in R1 or chance node)
    if community is None:
        return False

    # Round 2 not yet started (just saw the chance node) — not terminal
    if r2 is None:
        return False

    # Round 2 has ended
    return _r2_ended(r2)


def is_chance_node(history: str) -> bool:
    """
    Return True when round 1 has ended without a fold and the community card
    has not yet been dealt — i.e. the next step is a chance event, not a
    player decision.

    Examples
    --------
    ""      → False
    "bk"    → True   ← R1 over, no fold, community not yet dealt
    "bk/Q"  → False  ← community dealt, round 2 starts now
    "bk/Q/" → False  ← round 2 in progress
    """
    r1, community, _ = _split_history(history)
    return _r1_ended_no_fold(r1) and community is None


def whose_turn(history: str) -> int:
    """
    Return which player acts next: 0 = Player 1, 1 = Player 2.

    Leduc Poker uses the same first-player-acts-first convention as Kuhn.
    The acting player within a round is determined purely by the number
    of actions so far in that round:
        0 actions → Player 1 acts (starts each round)
        1 action  → Player 2 responds
        2 actions → Player 1 responds (only when facing a raise)

    This holds for both rounds.
    """
    ra = _round_actions(history)
    return len(ra) % 2   # 0 actions → P1, 1 action → P2, 2 actions → P1, …


def get_legal_actions(history: str) -> list[str]:
    """
    Return the list of legal actions at a non-terminal, non-chance history.

    Actions depend on:
      • Whether a bet/raise is currently facing the player
      • Whether the raise limit (1 per round) has been reached
    """
    if is_terminal(history):
        raise ValueError(f"No actions at terminal history '{history}'")
    if is_chance_node(history):
        raise ValueError(f"No player actions at chance node '{history}'")

    ra = _round_actions(history)

    if _bet_is_facing(ra):
        actions = [FOLD, CALL]
        if _raise_count(ra) < 1:   # at most 1 raise per round
            actions.append(RAISE)
        return actions
    else:
        return [CHECK, BET]


# ─────────────────────────────────────────────────────────────────────────────
#  Payoff computation
# ─────────────────────────────────────────────────────────────────────────────

def _pot_at_terminal(history: str) -> tuple[int, int]:
    """
    Return (p1_total_contributed, p2_total_contributed) at a terminal history.

    Both players ante 1 chip before play. Bets/calls/raises add chips.

    Round 1 bet size = 2, Round 2 bet size = 4.
    """
    r1, community, r2 = _split_history(history)

    p1 = 1  # ante
    p2 = 1  # ante

    def add_round_chips(actions: str, bet_size: int) -> tuple[int, int]:
        """Return chips added per player in this round's action sequence."""
        c1 = c2 = 0
        # track who is acting: P1 acts on even indices (0, 2, 4, ...)
        for idx, action in enumerate(actions):
            actor = idx % 2  # 0 = P1, 1 = P2
            if action in (BET, CALL, RAISE):
                if actor == 0:
                    c1 += bet_size
                else:
                    c2 += bet_size
        return c1, c2

    r1_p1, r1_p2 = add_round_chips(r1, BET_SIZE[1])
    p1 += r1_p1
    p2 += r1_p2

    if r2 is not None:
        r2_p1, r2_p2 = add_round_chips(r2, BET_SIZE[2])
        p1 += r2_p1
        p2 += r2_p2

    return p1, p2


def _has_pair(private_card: int, community_card: int) -> bool:
    return private_card == community_card


def terminal_payoff(
    history        : str,
    p1_card        : int,
    p2_card        : int,
    community_card : int,
) -> float:
    """
    Compute Player 1's payoff at a terminal history.

    Parameters
    ----------
    history        : terminal history string
    p1_card        : Player 1's private card rank (JACK=0 / QUEEN=1 / KING=2)
    p2_card        : Player 2's private card rank
    community_card : the face-up community card rank (only needed for showdown)

    Returns
    -------
    float — chips won (positive) or lost (negative) by Player 1.
    """
    assert is_terminal(history), f"'{history}' is not terminal"

    p1_contrib, p2_contrib = _pot_at_terminal(history)
    r1, community, r2 = _split_history(history)

    # ── Fold outcomes ──────────────────────────────────────────────────────
    # Determine who folded and in which round
    folded_in_r1 = _r1_ended_with_fold(r1)

    if folded_in_r1:
        # Last action in r1 is 'f'; the folder is the player who acted last
        folder = (len(r1) - 1) % 2   # 0 = P1 folded, 1 = P2 folded
        if folder == 0:
            return -p1_contrib         # P1 folded, loses what they put in
        else:
            return p2_contrib          # P2 folded, P1 wins P2's chips

    # r2 ended with a fold
    if r2 is not None and r2 in _R2_FOLD_TERMINALS:
        folder = (len(r2) - 1) % 2
        if folder == 0:
            return -p1_contrib
        else:
            return p2_contrib

    # ── Showdown ───────────────────────────────────────────────────────────
    p1_pair = _has_pair(p1_card, community_card)
    p2_pair = _has_pair(p2_card, community_card)

    if p1_pair and not p2_pair:
        p1_wins = True
    elif p2_pair and not p1_pair:
        p1_wins = False
    else:
        # Neither or both have pairs — higher rank wins
        p1_wins = p1_card > p2_card

    return p2_contrib if p1_wins else -p1_contrib


# ─────────────────────────────────────────────────────────────────────────────
#  Infoset key
# ─────────────────────────────────────────────────────────────────────────────

def infoset_key(private_card: int, community_card: int | None, history: str) -> str:
    """
    Build the information set key used to index into CFR nodes.

    Parameters
    ----------
    private_card   : the acting player's private card rank
    community_card : the face-up community card (None before it is dealt)
    history        : the full game history string

    Returns
    -------
    str — e.g. "K/-:b",  "K/J:bk/J/c"

    Format: "<private_rank>/<community_or_->:<history>"
    The community slot shows '-' before the community card is dealt.
    """
    comm_str = CARD_NAMES[community_card] if community_card is not None else "-"
    return f"{CARD_NAMES[private_card]}/{comm_str}:{history}"


# ─────────────────────────────────────────────────────────────────────────────
#  LeducPoker class
# ─────────────────────────────────────────────────────────────────────────────

class LeducPoker:
    """
    Leduc Poker game environment.

    Provides the same interface as KuhnPoker so that LeducCFRTrainer can
    reuse the two-pass CFR design with minimal changes.  Key extensions:

      • get_all_private_deals() — 30 ordered (p1_idx, p2_idx) pairs drawn
            from the 6-card indexed deck; CFR weights each deal uniformly
            (1/30), which correctly captures the true deal distribution
            (same-rank pairs appear twice, giving them double the weight).
      • get_community_cards(p1_idx, p2_idx) — posterior over the 4 remaining
            cards given which two physical cards were already dealt
      • is_chance_node(history) — True when R1 ended without a fold
      • terminal_payoff requires the community card rank as an extra argument
    """

    def __init__(self):
        # All 30 ordered draws of 2 cards from the 6-card indexed deck.
        # Each element is (p1_card_index, p2_card_index).
        # DECK[index] gives the rank.  CFR iterates over these uniformly.
        self.deals: list[tuple[int, int]] = list(permutations(range(6), 2))

    # ── Public interface (mirrors KuhnPoker) ──────────────────────────────────

    def is_terminal(self, history: str) -> bool:
        return is_terminal(history)

    def is_chance_node(self, history: str) -> bool:
        return is_chance_node(history)

    def whose_turn(self, history: str) -> int:
        return whose_turn(history)

    def get_legal_actions(self, history: str) -> list[str]:
        return get_legal_actions(history)

    def get_payoff(
        self,
        history        : str,
        p1_card        : int,
        p2_card        : int,
        community_card : int,
    ) -> float:
        """Player 1's payoff at a terminal node. Arguments are card *ranks*."""
        return terminal_payoff(history, p1_card, p2_card, community_card)

    def get_infoset_key(
        self,
        private_card   : int,
        community_card : int | None,
        history        : str,
    ) -> str:
        """private_card and community_card are *ranks* (JACK/QUEEN/KING)."""
        return infoset_key(private_card, community_card, history)

    # ── Deals and community cards ─────────────────────────────────────────────

    def get_all_private_deals(self) -> list[tuple[int, int]]:
        """
        Return all 30 ordered (p1_card_index, p2_card_index) pairs.

        Card indices refer to positions in the 6-card deck
        [J, J, Q, Q, K, K].  Two deals with the same rank combination
        (e.g. indices (0,1) and (1,0), both JJ) appear separately so that
        CFR's uniform weighting over this list is equivalent to the true
        deal probability:

            P(same-rank pair) = 2/30 per ordered pair  (e.g. JJ appears as
                                (0,1) and (1,0) — 2 deals out of 30)
            P(mixed-rank pair) = 4/30 per ordered rank pair  (e.g. JQ appears
                                as (0,2), (0,3), (1,2), (1,3) — 4 deals)

        To recover the rank of a card from its index: DECK[index].

        Deal probability table:
            Same-rank pair  (JJ, QQ, KK): 2 physical orderings × (1/30) = 2/30 each
            Mixed-rank pair (JQ, JK, …):  4 physical orderings × (1/30) = 4/30 each
            → same-rank pairs are half as likely as mixed-rank pairs  ✓
        """
        return self.deals

    def get_community_cards(
        self, p1_idx: int, p2_idx: int
    ) -> list[tuple[int, float]]:
        """
        Return the possible community card draws after both private cards
        have been dealt, as a list of (rank, probability) pairs.

        Given that card indices p1_idx and p2_idx are already dealt,
        4 cards remain in the deck.  Each is equally likely to be the
        community card (prob = 1/4).  We aggregate by rank and sum
        probabilities for any rank that appears more than once among the
        remaining cards.

        Parameters
        ----------
        p1_idx, p2_idx : int
            Card *indices* (0–5) of each player's private card.

        Returns
        -------
        list of (rank, probability) tuples — length 2 or 3 depending on
        how many distinct ranks remain.
        """
        dealt = {p1_idx, p2_idx}
        remaining_ranks: list[int] = [
            DECK[i] for i in range(6) if i not in dealt
        ]
        total = len(remaining_ranks)   # always 4

        # Aggregate probabilities by rank
        rank_prob: dict[int, float] = {}
        for rank in remaining_ranks:
            rank_prob[rank] = rank_prob.get(rank, 0.0) + 1.0 / total

        return list(rank_prob.items())


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    game = LeducPoker()

    print("=" * 60)
    print("Leduc Poker Environment — Sanity Checks")
    print("=" * 60)

    # ── Check 1: private deals ────────────────────────────────────────────────
    print("\n[1] Total ordered deals from 6-card deck:")
    deals = game.get_all_private_deals()
    print(f"    {len(deals)} deals  (expected 30 = 6×5)")
    rank_pair_counts: dict[tuple, int] = {}
    for p1i, p2i in deals:
        rp = (DECK[p1i], DECK[p2i])
        rank_pair_counts[rp] = rank_pair_counts.get(rp, 0) + 1
    print("    Same-rank pairs appear 2×, mixed-rank pairs appear 4×:")
    for pair, cnt in sorted(rank_pair_counts.items()):
        names = f"({CARD_NAMES[pair[0]]},{CARD_NAMES[pair[1]]})"
        print(f"      {names}: {cnt} deals")
    assert len(deals) == 30, "Expected 30 deals"

    # ── Check 2: community card probabilities ─────────────────────────────────
    print("\n[2] Community card distributions (sample deals by index):")
    # JJ: indices (0,1) — only Q and K remain (2 each)
    options = game.get_community_cards(0, 1)
    print(f"    P1=J(0) P2=J(1): ", end="")
    print(", ".join(f"{CARD_NAMES[r]}={p:.3f}" for r, p in options))
    assert abs(sum(p for _, p in options) - 1.0) < 1e-9
    # JQ: indices (0,2) — deck minus J[0] and Q[2] → J,Q,K,K remain
    options = game.get_community_cards(0, 2)
    print(f"    P1=J(0) P2=Q(2): ", end="")
    print(", ".join(f"{CARD_NAMES[r]}={p:.3f}" for r, p in options))
    assert abs(sum(p for _, p in options) - 1.0) < 1e-9
    print("    Probability sums verified ✓")

    # ── Check 3: terminal detection ───────────────────────────────────────────
    print("\n[3] Terminal history detection:")
    terminal_cases = [
        ("bf",        True),
        ("cbrk/Q/cc", True),
        ("bk/J/bk",   True),
        ("bk",        False),
        ("bk/Q/",     False),
        ("bk/Q/b",    False),
        ("",          False),
    ]
    all_ok = True
    for h, expected in terminal_cases:
        result = game.is_terminal(h)
        status = "✓" if result == expected else "✗"
        print(f"    {status} is_terminal('{h}') = {result}  (expected {expected})")
        if result != expected:
            all_ok = False
    if all_ok:
        print("    All terminal checks passed ✓")

    # ── Check 4: chance node detection ───────────────────────────────────────
    print("\n[4] Chance node detection:")
    chance_cases = [
        ("cc",   True),
        ("bk",   True),
        ("brk",  True),
        ("cbk",  True),
        ("",     False),
        ("b",    False),
        ("bk/Q", False),
    ]
    all_ok = True
    for h, expected in chance_cases:
        result = game.is_chance_node(h)
        status = "✓" if result == expected else "✗"
        print(f"    {status} is_chance_node('{h}') = {result}  (expected {expected})")
        if result != expected:
            all_ok = False
    if all_ok:
        print("    All chance node checks passed ✓")

    # ── Check 5: legal actions ────────────────────────────────────────────────
    print("\n[5] Legal actions at sample histories:")
    action_cases = [
        ("",        [CHECK, BET]),
        ("b",       [FOLD, CALL, RAISE]),
        ("br",      [FOLD, CALL]),
        ("bk/Q/",   [CHECK, BET]),
        ("bk/Q/b",  [FOLD, CALL, RAISE]),
        ("bk/Q/br", [FOLD, CALL]),
    ]
    for h, expected in action_cases:
        actions = game.get_legal_actions(h)
        status = "✓" if actions == expected else "✗"
        print(f"    {status} get_legal_actions('{h}') = {actions}")

    # ── Check 6: payoffs ──────────────────────────────────────────────────────
    print("\n[6] Payoff checks (using ranks directly):")

    pay = game.get_payoff("bf", JACK, KING, QUEEN)
    print(f"    bf  (P2 folds R1):             P1 payoff = {pay}  (expected +1)")
    assert pay == 1

    pay = game.get_payoff("brf", JACK, KING, QUEEN)
    print(f"    brf (P1 folds facing raise):   P1 payoff = {pay}  (expected -3)")
    assert pay == -3

    pay = game.get_payoff("cc/J/cc", JACK, QUEEN, JACK)
    print(f"    cc/J/cc (P1 pair vs P2 none):  P1 payoff = {pay}  (expected +1)")
    assert pay == 1

    pay = game.get_payoff("cc/J/cc", QUEEN, JACK, JACK)
    print(f"    cc/J/cc (P2 pair vs P1 none):  P1 payoff = {pay}  (expected -1)")
    assert pay == -1

    pay = game.get_payoff("cc/Q/cc", KING, JACK, QUEEN)
    print(f"    cc/Q/cc (K vs J, no pair):     P1 payoff = {pay}  (expected +1)")
    assert pay == 1

    print("    All payoff assertions passed ✓")

    # ── Check 7: infoset keys ─────────────────────────────────────────────────
    print("\n[7] Infoset key examples:")
    examples = [
        (KING,  None,  "b"),
        (QUEEN, JACK,  "bk/J/"),
        (JACK,  QUEEN, "cc/Q/br"),
    ]
    for private, comm, hist in examples:
        key = game.get_infoset_key(private, comm, hist)
        print(f"    private={CARD_NAMES[private]} comm={CARD_NAMES[comm] if comm is not None else '-'}"
              f" history='{hist}' → key='{key}'")

    print("\n" + "=" * 60)
    print("Phase 4 environment verified. Ready for LeducCFRTrainer.")
    print("=" * 60)
