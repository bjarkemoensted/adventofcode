# ·   ·  `*·.. · *  .`+· · `*. ··      · ·.` · .+·*  ·  .  `  · *.  ` ·   .·.* ·
# *.+·  ·.·`     ·   *·` *.  · Rock Paper Scissors    .·`  ·. •· +··.  ·  `•· ·.
# .··`• . ·*    . ·`*· https://adventofcode.com/2022/day/2 . +·*· `.·  `+*· `·.*
# ·`.·  *•.··   ·` .· * ·  ·     ·*.  ··     .+·`. ·  *. `· * .·  ·     ·  . .· 


def parse(s: str) -> list[tuple[str, str]]:
    res = []
    for line in s.splitlines():
        a, b = line.split()
        res.append((a, b))

    return res


# Map a move (ABC or XYZ) to a number 0, 1, 2, representing rock, paper, or scissors
map_: dict[str, int] = dict(zip("ABCXYZ", [0, 1, 2, 0, 1, 2]))


def get_points(moves: tuple[str, str]) -> int:
    """Compute points for regular rock paper scissors game"""

    opponent, player = (map_[s] for s in moves)
    res = 0
    res += player + 1

    draw = opponent == player
    won = player == (opponent + 1) % 3
    if draw:
        res += 3
    elif won:
        res += 6

    return res


def get_points_ultra_secret(entry: tuple[str, str]) -> int:
    """Computes points after decrypting the top secret strategy guide"""
    opponent_str, outcome = entry
    opponent = map_[opponent_str]

    res = 0
    draw = outcome == "Y"
    win = outcome == "Z"
    player_move = None
    if draw:
        res += 3
        player_move = opponent
    elif win:
        res += 6
        player_move = (opponent + 1) % 3
    else:
        player_move = (opponent + 2) % 3

    res += player_move + 1

    return res


def solve(data: str) -> tuple[int|str, ...]:
    guide = parse(data)

    star1 = sum(map(get_points, guide))
    print(f"Solution to part 1: {star1}")
    
    star2 = sum(map(get_points_ultra_secret, guide))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 2
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
