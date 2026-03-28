# ··* `· .·     ·` .+  · .`·  ·   ·`+· .·   * ··      ·*. `·   `·• ·      .·`·*·
# ·.`··*   .   ·`.  *··   ·`. ·   * Trick Shot  *  · · . ·  · *·•.` ·   *·  ·.`.
# *`·   ·  *· ·  ·  `  https://adventofcode.com/2021/day/17   ·    .*  ` .·+·* ·
# .··`.    ·  .`+ ·   *.·  *`·* . `·· `+·   ·.  ·  ``  ·   *+ . · ·` *· ·` `.`·*

import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> list[tuple[int, int]]:
    # target area: x=269..292, y=-68..-44
    stuff = s.split("target area: ")[1].split(", ")
    res = []
    for s in stuff:
        a, b = tuple(int(x) for x in s.split("=")[1].split(".."))
        res.append((a, b))
    return res


def within_borders(coords: NDArray[np.int_], borders: list[tuple[int, int]]) -> bool:
    """Determines whether input coordinates fall within target area"""
    for coord, (a, b) in zip(coords, borders):
        if not (a <= coord <= b):
            return False
        #
    return True


def missed(coords: NDArray[np.int_], borders: list[tuple[int, int]]) -> bool:
    """Determines if a shot has missed the target. It has missed if it's too far right or down,
    because the probe can never go left or up."""
    return coords[0] > borders[0][1] or coords[1] < borders[1][0]


class Probe:
    """Represents the position and velocity of a probe"""

    def __init__(self, pos: list[int], vel: list[int]):
        self.pos = np.array(pos)
        self.vel = np.array(vel)

    def tick(self) -> None:
        """Updates the probe's position and velocity vectors."""
        self.pos += self.vel
        a, b = self.vel
        # y-velocity decrements by one because gravity
        b -= 1
        # x-velocity approaches 0 because friction/resistance
        if a != 0:
            a -= int(a > 0)
        self.vel = np.array([a, b])


def determine_final_delta_x(dx: int) -> int:
    """The final horizontal distance is x + (x - 1) + ... = x(x+1)/2 (the Gauss trick)"""
    res = dx*(dx + 1)//2
    return res


def determine_min_vx(target: int) -> int:
    """Som initial x-velocities will never reach the target.
    This finds the minimum x-velocity such that the probe doesn't stall
    before reaching target."""
    
    running = 0
    while determine_final_delta_x(running) < target:
        running += 1

    return running


def determine_max_vy(target: int):
    """After the probe comes back down to y=0, if the vertical speed is so great the probe
    passes through the target, we've exceeded the max y-velocity."""
    vy = -target
    return vy


def trace_probe(vx: int, vy: int, borders: list[tuple[int, int]]) -> NDArray[np.int_]:
    """Trace every position visited by a probe fired with the specified initial velocity, until it either
    enters the target zone, or passes it"""

    origin = [0, 0]
    probe = Probe(pos=origin, vel=[vx, vy])
    # While probe isn't in the target, update its position
    positions = [probe.pos.copy()]
    while not within_borders(probe.pos, borders):
        probe.tick()
        if missed(probe.pos, borders):
            break
        positions.append(probe.pos.copy())
    
    # Collect positions in a single 2D array
    res = np.vstack(positions)
    return res


def solve(data: str) -> tuple[int|str, ...]:
    borders = parse(data)
    xlim, ylim = borders

    # Determine the parameter space to search
    min_vx = determine_min_vx(target=xlim[0])
    max_vx = xlim[1] + 1
    min_vy = ylim[0] - 1
    max_vy = determine_max_vy(ylim[0])

    # Get trajectories of all probes hitting the target
    trajectories = [trace_probe(vx, vy, borders) for vx in range(min_vx, max_vx+1) for vy in range(min_vy, max_vy+1)]
    hits = [tr for tr in trajectories if within_borders(tr[-1], borders)]

    # Get the max y-value for the target-hitting probes
    all_positions = np.vstack(hits)
    star1 = all_positions[:, 1].max().item()
    print(f"Solution to part 1: {star1}")

    star2 = len(hits)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 17
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
