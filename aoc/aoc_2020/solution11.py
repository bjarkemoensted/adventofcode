# ·.`*   · *.·` ·     •··      ·.·*•`. ` ·+· *   ·.·`  ·   *·. ·  ·`  +   *·.`··
# ·`*` ·   ·*` •  . · ·`   .`·*   Seating System .+·    •  ·.*  ·· ·   .·  .  *`
# .*··•` ··.`*     · + https://adventofcode.com/2020/day/11 *`·. *  .·   •·  · .
# *·. ` . * · .·  ·* `   ·*·.     `·  • ·   .`.··  `·* •.·      `. .·* · ·`*  .·

import numpy as np
from numpy.typing import NDArray
import typing as t


_floor = "."
_seat = "L"
_occupied = "#"

raw = """L.LL.LL.LL
LLLLLLL.LL
L.L.L..L..
LLLL.LL.LL
L.LL.LL.LL
L.LLLLL.LL
..L.L.....
LLLLLLLLLL
L.LLLLLL.L
L.LLLLL.LL"""


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line.strip()) for line in s.splitlines()])
    return res


class Simulation:
    def __init__(self, map_: NDArray[np.str_]) -> None:
        self.map_ = map_
        
        self.coords_flattened = np.zeros(shape=self.map_.shape, dtype=int)
        self.inds = []
        for ind, (i, j) in enumerate(np.ndindex(self.map_.shape)):
            self.coords_flattened[i, j] = ind
            self.inds.append((i, j))
        
        #
    
    def visualize_state(self, state: NDArray[np.int_]) -> str:
        viz = self.map_.copy()
        for ind, val in enumerate(state):
            if val:
                coord = self.inds[ind]
                if viz[*coord] != _seat:
                    raise RuntimeError
                viz[*coord] = _occupied
            #
        
        lines = ["".join(row) for row in viz]
        res = "\n".join(lines)
        return res




def solve(data: str) -> tuple[int|str, ...]:
    map_ = parse(data)

    sim = Simulation(map_)

    star1 = -1
    print(f"Solution to part 1: {star1}")

    star2 = -1
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 11
    from aocd import get_data
    #raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
