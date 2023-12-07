from collections import Counter
from functools import cache


def read_input():
    with open("input07.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = []
    for line in s.split("\n"):
        hand, bid_s = line.split()
        bid = int(bid_s)
        res.append((hand, bid))

    return res


def _count_cards(hand):
    """Returns a sorted tuple of the number of cards in a hand e.g. (3, 1, 1) for 3 of a kind"""
    counts = tuple(sorted(Counter(hand).values(), reverse=True))
    return counts


@cache
def type_strength(counts):
    """Determines the type strength of a hand. Higher values for stronger types."""
    types = [
        (5,),  # 5 of a kind
        (4, 1),  # 4 of a kind
        (3, 2),  # full house
        (3, 1, 1),  # 3 of a kind
        (2, 2, 1),  # 2 pairs
        (2, 1, 1, 1),  # pair
        (1, 1, 1, 1, 1)  # high card
    ]

    res = None
    for i, type_ in enumerate(types[::-1]):
        if counts == type_:
            res = i
            break
        #

    return res


@cache
def card_strength(c, joker=False):
    vals = "23456789TJQKA" if not joker else "J23456789TQKA"
    d = {char: i for i, char in enumerate(vals)}
    return d[c]


def hand_strength(hand, joker=False):
    if joker:
        # If playing with the joker rule, the joker imitates the most common card in the hand
        d = Counter(hand)
        try:
            most_common_card = max((k for k in d.keys() if k != "J"), key=lambda k: d[k])
        except ValueError:
            # If the hand is all jokers, it makes no difference which card the jokers imitate, just use aces.
            most_common_card = "A"
        counts = _count_cards(hand.replace("J", most_common_card))
    else:
        counts = _count_cards(hand)
    ts = type_strength(counts)
    strength = tuple([ts] + [card_strength(card, joker=joker) for card in hand])

    return strength


def compute_total_winnings(hand_bid_tuples, joker=False):
    ordered_tups = sorted(hand_bid_tuples, key=lambda t: hand_strength(t[0], joker=joker))
    res = sum((i+1)*bid for i, (_, bid) in enumerate(ordered_tups))
    return res


def main():
    raw = read_input()
    parsed = parse(raw)

    star1 = compute_total_winnings(parsed)
    print(f"Total winnings: {star1}.")

    star2 = compute_total_winnings(parsed, joker=True)
    print(f"Total winnings with joker rule: {star2}.")


if __name__ == '__main__':
    main()
