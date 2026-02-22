# . ·`.` ·* · ··`  ·  `   ·*  .··   ·    · `+ . * ··.  +`*·    ·  ·.*`· *.·  ·`·
# `  · *.` · . +·  .  *··`. ·*  + Combo Breaker  · *·`  .·`*·.      ·· *`+.·•.·`
# *·· `  . · ·. *   *· https://adventofcode.com/2020/day/25  `· * *   ··  + ` .·
# ·`.+· * ·`. ·  • *· ·`.• · `·  · * `· . +· ·* ·` .`+   .*· ·  +·    *  ·`.· ·*

import numba

_mod = 20201227


def parse(s: str) -> tuple[int, int]:
    door_public, card_public = map(int, s.splitlines())
    return door_public, card_public


@numba.njit(cache=True)
def get_loop_size(target_value: int, subject_number: int=7):
    running = 1
    loop = 0
    while running != target_value:
        running = (running * subject_number) % _mod
        loop += 1
    return loop


numba.njit(cache=True)
def transform(subject_number: int, loop_size: int):
    res = 1
    for _ in range(loop_size):
        res = (res * subject_number) % _mod
    return res


def solve(data: str) -> tuple[int|str, None]:
    door_public, card_public = parse(data)

    card_loop_size = get_loop_size(target_value=card_public)
    encryption_key = transform(door_public, card_loop_size)

    star1 = encryption_key
    print(f"Solution to part 1: {star1}")

    return star1, None


def main() -> None:
    year, day = 2020, 25
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
