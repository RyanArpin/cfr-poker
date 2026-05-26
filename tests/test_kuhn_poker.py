"""
tests/test_kuhn_poker.py
────────────────────────
Formal pytest test suite for the Kuhn Poker game environment (Phase 1).

Run from the project root:
    pytest tests/test_kuhn_poker.py -v

These tests verify:
  1. Card constants and naming
  2. Terminal history detection
  3. Turn order (whose_turn)
  4. Legal actions
  5. Payoffs — all terminal histories, all 6 deals
  6. Infoset key format
  7. Deal enumeration (exactly 6 deals, all unique)
  8. Game tree structure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from kuhn.kuhn_poker import (
    KuhnPoker,
    JACK, QUEEN, KING,
    CARD_NAMES,
    is_terminal,
    whose_turn,
    get_legal_actions,
    terminal_payoff,
    infoset_key,
    TERMINAL_HISTORIES,
)

# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def game():
    return KuhnPoker()

# ─── 1. Card constants ─────────────────────────────────────────────────────────

class TestCardConstants:
    def test_ordering(self):
        """King beats Queen beats Jack."""
        assert JACK < QUEEN < KING

    def test_names(self):
        assert CARD_NAMES[JACK]  == "J"
        assert CARD_NAMES[QUEEN] == "Q"
        assert CARD_NAMES[KING]  == "K"

# ─── 2. Terminal history detection ────────────────────────────────────────────

class TestIsTerminal:
    def test_terminal_histories(self):
        for h in ["cc", "bc", "bf", "cbc", "cbf"]:
            assert is_terminal(h), f"'{h}' should be terminal"

    def test_non_terminal_histories(self):
        for h in ["", "c", "b", "cb"]:
            assert not is_terminal(h), f"'{h}' should NOT be terminal"

    def test_terminal_set_is_complete(self):
        """There are exactly 5 terminal histories in Kuhn Poker."""
        assert len(TERMINAL_HISTORIES) == 5

# ─── 3. Turn order ────────────────────────────────────────────────────────────

class TestWhoseTurn:
    def test_empty_history_is_p1(self):
        assert whose_turn("") == 0

    def test_after_check_is_p2(self):
        assert whose_turn("c") == 1

    def test_after_bet_is_p2(self):
        assert whose_turn("b") == 1

    def test_after_check_bet_is_p1(self):
        """P1 checked, P2 bet, now P1 must respond."""
        assert whose_turn("cb") == 0

# ─── 4. Legal actions ─────────────────────────────────────────────────────────

class TestLegalActions:
    def test_opening_actions(self):
        """P1 can check or bet from an empty history."""
        assert set(get_legal_actions("")) == {"c", "b"}

    def test_after_check(self):
        """P2 can check or bet after P1 checks."""
        assert set(get_legal_actions("c")) == {"c", "b"}

    def test_after_bet(self):
        """P2 can only fold or call after P1 bets (no re-raise in Kuhn)."""
        assert set(get_legal_actions("b")) == {"f", "c"}

    def test_after_check_bet(self):
        """P1 can only fold or call after check-bet."""
        assert set(get_legal_actions("cb")) == {"f", "c"}

    def test_no_actions_at_terminal(self):
        """Requesting actions at a terminal node should raise ValueError."""
        with pytest.raises(ValueError):
            get_legal_actions("cc")

# ─── 5. Payoffs ───────────────────────────────────────────────────────────────

class TestPayoffs:
    """
    Reference payoff table from Zinkevich et al. / Kuhn (1950).
    All payoffs from Player 1's perspective (zero-sum: P2 payoff = -P1 payoff).
    """

    # ── Fold outcomes (card-independent) ──────────────────────────────────────

    def test_bf_p1_always_wins_one(self, game):
        """After bet-fold, P1 wins +1 regardless of cards."""
        for p1, p2 in game.get_all_deals():
            assert game.get_payoff("bf", p1, p2) == 1

    def test_cbf_p1_always_loses_one(self, game):
        """After check-bet-fold, P1 loses -1 regardless of cards."""
        for p1, p2 in game.get_all_deals():
            assert game.get_payoff("cbf", p1, p2) == -1

    # ── Showdown with pot = 2 (both checked) ──────────────────────────────────

    def test_cc_king_beats_queen(self, game):
        assert game.get_payoff("cc", KING, QUEEN) == 1

    def test_cc_king_beats_jack(self, game):
        assert game.get_payoff("cc", KING, JACK) == 1

    def test_cc_queen_beats_jack(self, game):
        assert game.get_payoff("cc", QUEEN, JACK) == 1

    def test_cc_jack_loses_to_queen(self, game):
        assert game.get_payoff("cc", JACK, QUEEN) == -1

    def test_cc_jack_loses_to_king(self, game):
        assert game.get_payoff("cc", JACK, KING) == -1

    def test_cc_queen_loses_to_king(self, game):
        assert game.get_payoff("cc", QUEEN, KING) == -1

    # ── Showdown with pot = 4 (called a bet) ──────────────────────────────────

    def test_bc_winner_gets_two(self, game):
        """Called bet: winner gains 2, loser loses 2."""
        assert game.get_payoff("bc", KING, JACK)  ==  2
        assert game.get_payoff("bc", JACK, KING)  == -2
        assert game.get_payoff("bc", KING, QUEEN) ==  2
        assert game.get_payoff("bc", QUEEN, KING) == -2

    def test_cbc_same_as_bc(self, game):
        """Check-bet-call has same pot as bet-call."""
        for p1, p2 in game.get_all_deals():
            assert game.get_payoff("cbc", p1, p2) == game.get_payoff("bc", p1, p2)

    # ── Zero-sum check ─────────────────────────────────────────────────────────

    def test_zero_sum_property(self, game):
        """Payoffs must sum to zero (Kuhn Poker is zero-sum)."""
        # We can't get P2's payoff directly, but we can verify symmetry:
        # swapping P1 and P2 cards negates the payoff.
        for hist in ["cc", "bc", "cbc"]:  # showdown histories only
            for p1, p2 in game.get_all_deals():
                p1_payoff = game.get_payoff(hist, p1, p2)
                # If P1 and P2 swap seats, P1's payoff should be negated
                swapped  = game.get_payoff(hist, p2, p1)
                assert p1_payoff == -swapped, (
                    f"Zero-sum violated at history='{hist}' "
                    f"p1={CARD_NAMES[p1]} p2={CARD_NAMES[p2]}"
                )

# ─── 6. Infoset keys ──────────────────────────────────────────────────────────

class TestInfosetKey:
    def test_format(self):
        """Infoset key must be '<CardLetter>:<history>'."""
        assert infoset_key(KING,  "")   == "K:"
        assert infoset_key(QUEEN, "b")  == "Q:b"
        assert infoset_key(JACK,  "cb") == "J:cb"

    def test_all_six_decision_infosets(self, game):
        """
        Kuhn Poker has exactly 12 infoset nodes total across the tree:
        3 for P1 (at histories "", "cb")   — wait, let's count carefully.

        P1's infosets: at history "" → 3 (one per card) = 3 nodes
        P2's infosets: at histories "c" and "b" → 3×2 = 6 nodes
        P1's infosets: at history "cb" → 3 nodes
        Total: 12 infoset nodes.
        """
        infosets = set()
        for p1, p2 in game.get_all_deals():
            for history in ["", "cb"]:        # P1's decision points
                key = infoset_key(p1, history)
                infosets.add(key)
            for history in ["c", "b"]:        # P2's decision points
                key = infoset_key(p2, history)
                infosets.add(key)

        # 3 cards × 4 decision histories = 12 unique infoset keys
        assert len(infosets) == 12

# ─── 7. Deal enumeration ──────────────────────────────────────────────────────

class TestDeals:
    def test_exactly_six_deals(self, game):
        """With 3 cards and 2 players, there are 3!/(3-2)! = 6 ordered deals."""
        assert len(game.get_all_deals()) == 6

    def test_all_deals_unique(self, game):
        deals = game.get_all_deals()
        assert len(deals) == len(set(deals))

    def test_no_deal_has_same_card_twice(self, game):
        """Both players cannot hold the same card."""
        for p1, p2 in game.get_all_deals():
            assert p1 != p2

    def test_all_cards_appear_as_p1_and_p2(self, game):
        """Every card appears 2 times as P1's card and 2 times as P2's card."""
        from collections import Counter
        p1_cards = Counter(p1 for p1, _ in game.get_all_deals())
        p2_cards = Counter(p2 for _, p2 in game.get_all_deals())
        for card in [JACK, QUEEN, KING]:
            assert p1_cards[card] == 2
            assert p2_cards[card] == 2

# ─── 8. Game tree structure ───────────────────────────────────────────────────

class TestGameTree:
    def test_non_terminal_count(self, game):
        """
        Non-terminal histories: "", "c", "b", "cb" → exactly 4.
        """
        non_terminals = [h for h in game._all_histories() if not is_terminal(h)]
        assert len(non_terminals) == 4

    def test_terminal_count(self, game):
        """Terminal histories: "cc", "bc", "bf", "cbc", "cbf" → exactly 5."""
        terminals = [h for h in game._all_histories() if is_terminal(h)]
        assert len(terminals) == 5

    def test_total_histories(self, game):
        """Total histories = 4 non-terminal + 5 terminal = 9."""
        assert len(game._all_histories()) == 9