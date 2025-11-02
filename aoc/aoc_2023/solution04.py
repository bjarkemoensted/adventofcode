# `*. ·  ·`  . +·  ..·• `  * · .· +` . `· * ·.`  · * · .`•  `··  . +  ··• .`   ·
# *.··   `. • ·   ·`  ·  *     ·*· Scratchcards    ·    * ` ·       · . ` ·+· . 
# .·`. · ·  .* `    ·  https://adventofcode.com/2023/day/4  •  ` ·*.. ·  ·` *·•.
# ·   * .   · `· +.   `•·`.    · *` . ·. ·      ·   `·  .`·     +`· *·` .  ·`*·+


import re


def parse(s) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Parses into a list of tuples of winning cards and cards drawn"""
    games = []
    for line in s.split("\n"):
        m = re.match(r'Card\s*(\d+):\s*(.*)\|(.*)', line)
        assert m is not None
        matches = m.groups()
        _ = int(matches[0])
        winners, drawn = (tuple(int(elem) for elem in substring.strip().split()) for substring in matches[1:])
        games.append((winners, drawn))
    
    return games


def get_n_winning_numbers(winners: tuple[int, ...], drawn: tuple[int, ...]) -> int:
    """Returns the number of cards that are contained in the set of winning cards"""
    n = len(set(winners).intersection(set(drawn)))
    return n


def compute_points(winners: tuple[int, ...], drawn: tuple[int, ...]) -> int:
    n = get_n_winning_numbers(winners, drawn)
    res = 0 if n == 0 else 2**(n - 1)
    return res


def count_cards(games: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> list[int]:
    """Count the number of cards resulting from playing with the modified rules"""
    n_cards = [1 for _ in games]
    for i, (winners, drawn) in enumerate(games):
        n = get_n_winning_numbers(winners, drawn)
        for shift in range(i + 1, i + n + 1):
            try:
                n_cards[shift] += n_cards[i]
            except IndexError:
                pass
        #

    return n_cards


def solve(data: str) -> tuple[int|str, ...]:
    games = parse(data)

    points = [compute_points(winners, drawn) for winners, drawn in games]
    star1 = sum(points)
    print(f"Solution to part 1: {star1}")

    n_cards = count_cards(games)
    star2 = sum(n_cards)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 4
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
