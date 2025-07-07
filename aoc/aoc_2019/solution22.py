# `ꞏ. .ꞏ*⸳` .⸳  *`     *   +.⸳ꞏ ` + ⸳  .ꞏ⸳`+   `  +.⸳•⸳`   .*  ꞏ.` ⸳* ꞏ ⸳. *• ⸳ꞏ
# *.`+ ⸳ ꞏ*     .   ⸳` ꞏ+  `ꞏ *⸳ . Slam Shuffle  ⸳` *`.⸳+    ꞏ*   ⸳ .*⸳ꞏ   . ⸳` 
# .⸳+`ꞏ ⸳`.  +  ⸳•*` . https://adventofcode.com/2019/day/22  +.   `• ꞏ   *.⸳ꞏ• .
# `*      ꞏ ꞏ`⸳+ +.⸳ꞏ⸳   .  ⸳•`  ꞏ *. • +ꞏ ` .       ` *ꞏ⸳ *⸳.  `+ꞏ*      ꞏ`.` ⸳

from collections import deque
from copy import deepcopy
from typing import TypeAlias

deck: TypeAlias = deque[int]


test = """deal into new stack
cut -2
deal with increment 7
cut 8
cut -4
deal with increment 7
cut 3
deal with increment 9
deal with increment 3
cut -1"""


def parse(s):
    res = []
    for line in s.splitlines():
        if line == "deal into new stack":
            elem = ("new_stack", ())
        elif line.startswith("cut"):
            elem = ("cut", (int(line.split()[-1]),))
        elif line.startswith("deal"):
            elem = ("deal_with_increment", (int(line.split()[-1]),))
        res.append(elem)
    return res


def deal_with_increment(cards: deck, n: int) -> deck:
    rearranged = [-1 for _ in cards]
    for i, card in enumerate(cards):
        rearranged[(n*i) % len(rearranged)] = card
    
    res = deque(rearranged)
    return res


def shuffle(n_cards: int, instructions: list[tuple[str, tuple]]) -> deck:
    cards = deque(range(n_cards))
    
    for op, args in instructions:
        match op:
            case "new_stack":
                cards.reverse()
            case "cut":
                n = args[0]
                cards.rotate(-n)
            case "deal_with_increment":
                cards = deal_with_increment(cards, *args)
            case _:
                raise RuntimeError(f"Unrecognized op: {op}")
    
    return cards


def solve(data: str):
    instructions = parse(data)
    n_cards = 10007
    
    shuffled = shuffle(n_cards, instructions)
    
    star1 = shuffled.index(2019)
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2019, 22
    from aocd import get_data
    raw = get_data(year=year, day=day)
    #raw = test
    solve(raw)


if __name__ == '__main__':
    main()
