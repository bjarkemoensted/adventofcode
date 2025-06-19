# ꞏ+`*.⸳ + ꞏ `*.⸳+  .⸳ *. ⸳ꞏ `  ⸳ *+    ꞏ . `  •     .*`     ꞏ⸳ • .``ꞏ⸳.   •`⸳ꞏ.
#  .⸳`+ .  •`•.     `  ꞏ ⸳*.•⸳` *  Marble Mania   ⸳  `  ꞏ `+.`+  . ꞏ     `ꞏ.  ⸳*
# `⸳*.•ꞏ     .⸳      ꞏ https://adventofcode.com/2018/day/9 ꞏ +`⸳ ⸳ *ꞏ  `  .⸳ . .
# ` .*  `.⸳    +⸳`.ꞏ* `    • .   *⸳ꞏ ` .*`   •⸳ .ꞏ+ ⸳ `  ꞏ⸳ +  .`*  ` .⸳ .  ⸳ *ꞏ

from collections import defaultdict, deque
import re


def parse(s):
    m = re.match(r"(\d+) players; last marble is worth (\d+) points", s)
    n_players, max_points = map(int, m.groups())
    return n_players, max_points


def _iter_players(n: int):
    inds = [i+1 for i in range(n)]
    while True:
        yield from inds


def play_marble_game(n_players: int, n_marbles: int) -> dict[int, int]:
    """Play the elves' marble game. Return a dict containing the final score for each player"""
    
    # Double-ended queue with the 'current marble' located at the left end
    marbles: deque[int] = deque([0])
    players = _iter_players(n_players)
    points: dict[int, int] = defaultdict(int)
    
    for marble in range(1, n_marbles):
        player = next(players)
        if marble % 23 == 0:
            # Add points to the player when they hit a marble which is a multiple of 23
            points[player] += marble
            # Also grab the marble 7 spaces clockwise to current marble
            marbles.rotate(7)
            points[player] += marbles.popleft()
            continue
        
        # Insert new marble between marbles 1 and 2 spaces clockwise of current. Shift by 2 and insert
        marbles.rotate(-2)
        marbles.appendleft(marble)
        
        
    return points


def solve(data: str):
    n_players, max_points = parse(data)
    
    n_marbles = max_points + 1
    points = play_marble_game(n_players, n_marbles)
    star1 = max(points.values())
    print(f"Solution to part 1: {star1}")
    
    n_marbles2 = max_points*100 + 1
    points2 = play_marble_game(n_players, n_marbles2)
    star2 = max(points2.values())
    
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2018, 9
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    
    solve(raw)


if __name__ == '__main__':
    main()
