# `·+. `·*  ··`   · `*·.  · . ·    +.`·  `*·     `.· ·`    *·.    `·     · `·. ·
# ·  `·  .`· *•· ·  +`.·*  ·  No Time for a Taxicab   ·.`·  ` ·   ·     ·*·  ·.`
# +`·· .   ·*` .·    · https://adventofcode.com/2016/day/1 .  `*·   * ··` +  •··
# `·.  ·` *   · ` .·      ·.*   ·   `*`· .·   · *   ·` +  .·  *·   ·• `  . ·*·`.


import numpy as np


def parse(s: str):
    steps = s.strip().split(", ")
    res = []
    for step in steps:
        rotate = step[0]
        dist = int(step[1:])
        res.append((rotate, dist))

    return res


def rotate_direction(direction, rotation):
    """Rotates a direction vector left (L) or right "R" by 90 degrees."""
    rotation_matrices = {
        "R": np.array([[0, -1], [1, 0]]),
        "L": np.array([[0, 1], [-1, 0]])
    }

    M = rotation_matrices[rotation]
    res = M.dot(direction)
    return res


def trace_path(steps):
    location = np.array([0, 0])
    direction = np.array([1, 0])
    def to_tup(loc): return tuple(int(val) for val in loc)
    path = [to_tup(location)]

    for rotation, distance in steps:
        direction = rotate_direction(direction, rotation)
        for _ in range(distance):
            location += direction
            path.append(to_tup(location))

    return path


def manhatten_dist(vec):
    return sum(map(abs, vec))


def get_first_double_visit(path):
    seen = set([])
    for loc in path:
        if loc in seen:
            return loc
        seen.add(loc)



def solve(data: str) -> tuple[int|str, int|str]:
    instructions = parse(data)

    path = trace_path(instructions)
    star1 = manhatten_dist(path[-1])
    print(f"Solution to part 1: {star1}")

    first_doubly_visited = get_first_double_visit(path)
    star2 = manhatten_dist(first_doubly_visited)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2016, 1
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
