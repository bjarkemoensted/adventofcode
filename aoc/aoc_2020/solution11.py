# ·.`*   · *.·` ·     •··      ·.·*•`. ` ·+· *   ·.·`  ·   *·. ·  ·`  +   *·.`··
# ·`*` ·   ·*` •  . · ·`   .`·*   Seating System .+·    •  ·.*  ·· ·   .·  .  *`
# .*··•` ··.`*     · + https://adventofcode.com/2020/day/11 *`·. *  .·   •·  · .
# *·. ` . * · .·  ·* `   ·*·.     `·  • ·   .`.··  `·* •.·      `. .·* · ·`*  .·

from copy import deepcopy
import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line.strip()) for line in s.splitlines()])
    return res



def get_vicinity(map_, i, j):
    res = []
    nr, nc = map_.shape
    for ii in range(max(i-1, 0), min(i+2, nr)):
        for jj in range(max(j-1, 0), min(j+2, nc)):
            if ii == i and jj == j:
                continue
            res.append(map_[ii, jj])
        #
    return res


def get_new_seat_state(seat, vicinity, crowd=4):
    seat_is_empty = seat == "L"
    vicinity_is_free = all(s != "#" for s in vicinity)
    if seat_is_empty and vicinity_is_free:
        return "#"

    seat_is_occupied = seat == "#"
    vicinity_is_crowded = sum(s == "#" for s in vicinity) >= crowd
    if seat_is_occupied and vicinity_is_crowded:
        return "L"

    return seat


class Simulation():
    def __init__(self, map_, crowd, get_vicinity):
        self.map_ = deepcopy(map_)
        self.stable = False
        self.crowd = crowd
        self.get_vicinity = get_vicinity

    def tick(self):
        old_state = deepcopy(self.map_)
        nr, nc = self.map_.shape
        something_changed = False
        for i in range(nr):
            for j in range(nc):
                seat_current = old_state[i, j]
                vicinity = self.get_vicinity(old_state, i, j)
                seat_new = get_new_seat_state(seat_current, vicinity, crowd=self.crowd)
                if seat_current != seat_new:
                    something_changed = True
                self.map_[i, j] = seat_new
            #
        if not something_changed:
            self.stable = True
        #

    def run(self, max_steps=1000):
        n = 0
        while not self.stable and n < max_steps:
            print(n)
            self.tick()
            n += 1
        if not self.stable:
            print("*** Did not converge!")
        #

    def count_occupied(self):
        n_occupied = sum(val == "#" for val in self.map_.flat)
        return n_occupied
    #



def line_of_sight_vicinity(map_, i, j):
    directions = []
    for a in range(-1, 2):
        for b in range(-1, 2):
            if not a == b == 0:
                directions.append((a, b))
            #
        #

    res = []
    for a, b in directions:
        ii, jj = i, j
        scan = True
        while scan:
            ii += a
            jj += b
            falloff = not all(0 <= ind < lim for ind, lim in zip((ii, jj), map_.shape))
            if falloff:
                break
            seat = map_[ii, jj]
            if seat in ("#", "L"):
                res.append(seat)
                scan = False
            #
        #
    return res


def solve(data: str) -> tuple[int|str, ...]:
    map_ = parse(data)

    sim = Simulation(map_, crowd=4, get_vicinity=get_vicinity)
    sim.run()
    star1 = sim.count_occupied()
    print(f"Solution to part 1: {star1}")

    newsim = Simulation(map_, crowd=5, get_vicinity=line_of_sight_vicinity)
    newsim.run()
    star2 = newsim.count_occupied()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 11
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
