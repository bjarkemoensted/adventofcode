# *   ·.`* ·     ·  · .   *. ` · .· `•· · +` •.`· .·   + . •·*·`· .  *·    *·.`·
# .`· + *.· `+·      .`·   ·    · Crossed Wires   *+·   `·   . . · * ` ·+ ·. · `
# `· ·   ·.  ` .·  +`* https://adventofcode.com/2019/day/3   `  ·+  ·   `    ·.·
# ·.` .·  +. ·`   ·  ·     `··`+     ·* ` ·.  ·  +    ·`• `· ·.  `  +·.  ·.·+* .

from dataclasses import dataclass
from typing import Iterator


@dataclass
class Wire:
    steps: tuple[tuple[str, int], ...]

    def iter_points(self) -> Iterator[tuple[int, int]]:
        """Iterate over all the points on the wire"""
        x = 0
        y = 0
        yield x, y

        for dir_, n in self.steps:
            # Determine step vector
            dx, dy = 0, 0
            match dir_:
                case "U":
                    dy += 1
                case "D":
                    dy -= 1
                case "R":
                    dx += 1
                case "L":
                    dx -= 1
                case _:
                    raise ValueError
                #
            
            # Generate the points until the next turn
            for _ in range(n):
                x += dx
                y += dy
                yield x, y
            #
        #
    #


def parse(s: str) -> tuple[Wire, Wire]:
    wires = []
    for line in s.splitlines():
        steps = tuple((s[0], int(s[1:])) for s in line.split(","))
        wires.append(Wire(steps=steps))
        
    first, second = wires
    return first, second


def fewest_combined_steps(wire_a: Wire, wire_b: Wire, targets: set[tuple[int, int]]) -> int:
    """Determines the fewest number of steps in which the two wires reach one of the targets"""
    
    # Assume infinitely many steps initially
    first_reached = {point: [float("inf"), float("inf")] for point in targets}

    # Determine when each wire reaches each target
    for i, wire in enumerate((wire_a, wire_b)):
        for n, p in enumerate(wire.iter_points()):
            if p in first_reached and first_reached[p][i] == float("inf"):
                first_reached[p][i] = n
            #
        #

    # Keep the smallest sum
    res = min(map(sum, first_reached.values()))
    assert isinstance(res, int)
    return res


def solve(data: str) -> tuple[int|str, ...]:
    wires = parse(data)
    
    # Take the min Manhatten dist to the intersection points (excluding origin)
    intersections = set.intersection(*map(set, (w.iter_points() for w in wires)))
    intersections.remove((0, 0))
    star1 = min(sum(map(abs, coords)) for coords in intersections)
    print(f"Solution to part 1: {star1}")

    star2 = fewest_combined_steps(*wires, intersections)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 3
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
