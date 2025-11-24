# · *·. ·`.  ·+*.·     `   ·`.    * ·+· `.    ·  .    ·*·*`  ·.  . * +· `· • `·.
# .·`*   ·+.+`· ·.   **·`    · `·  Camel Cards ` · *       .   ·+·*  . ·. `.··**
# ` ·   *. ·· ` • * .· https://adventofcode.com/2023/day/7      · .·  + · *· +·.
# *·.·`   ·*` .·`• ·· . ·. * ` + · ` · *+. · . ··*`    ·+ .   ·* •·.`·  *     .·

from collections import Counter
from functools import cache


def parse(s: str) -> list[tuple[str, int]]:
    res = []
    for line in s.split("\n"):
        hand, bid_s = line.split()
        bid = int(bid_s)
        res.append((hand, bid))

    return res


def _count_cards(hand: str) -> tuple[int, ...]:
    """Returns a sorted tuple of the number of cards in a hand e.g. (3, 1, 1) for 3 of a kind"""
    counts = tuple(sorted(Counter(hand).values(), reverse=True))
    return counts


@cache
def type_strength(counts: tuple[int, ...]) -> int:
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

    for i, type_ in enumerate(types[::-1]):
        if counts == type_:
            return i
        #
    
    raise RuntimeError("Unable to determine hand type")


@cache
def card_strength(card: str, joker=False) -> int:
    """Determines the strength of a single card"""
    
    vals = "23456789TJQKA" if not joker else "J23456789TQKA"
    res = next(i for i, char in enumerate(vals) if char == card)

    return res


def hand_strength(hand, joker=False) -> tuple[int, ...]:
    """Determine the strength of a hand"""
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


def compute_total_winnings(hand_bid_tuples, joker=False) -> int:
    ordered_tups = sorted(hand_bid_tuples, key=lambda t: hand_strength(t[0], joker=joker))
    res = sum((i+1)*bid for i, (_, bid) in enumerate(ordered_tups))
    return res


def solve(data: str) -> tuple[int|str, ...]:
    parsed = parse(data)

    star1 = compute_total_winnings(parsed)
    print(f"Solution to part 1: {star1}")

    star2 = compute_total_winnings(parsed, joker=True)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 7
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
