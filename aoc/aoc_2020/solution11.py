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
        
        # Assign a single index to each matrix element, so we can represent states as vectors
        self.coords_flattened = np.zeros(shape=self.map_.shape, dtype=int)
        self.inds = []
        can_change = []
        
        for ind, ((i, j), char) in enumerate(np.ndenumerate(self.map_)):
            self.coords_flattened[i, j] = ind
            self.inds.append((i, j))
            # store the non-floor inds so we can filter out the sites that may change
            can_change.append(int(char != _floor))
        
        self.seat_filter = np.array(can_change)
        self.n_elems = len(self.inds)
    
    def vicinity_transition_matrix(self) -> NDArray[np.int_]:
        M = np.zeros(shape=(self.n_elems, self.n_elems), dtype=int)
        span = (-1, 0, +1)
        dirs = [(di, dj) for di in span for dj in span if not di == dj == 0]

        for vec_i, (i, j) in enumerate(self.inds):
            for di, dj in dirs:
                xp = (i+di, j+dj)
                in_bounds = all(0 <= coord < lim for coord, lim in zip(xp, self.map_.shape))
                if in_bounds:
                    vec_j = self.coords_flattened[*xp]
                    M[vec_i, vec_j] = 1
                #
            #

        return M
    
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

    def ascii_to_state(self, map_: NDArray[np.str_]) -> NDArray[np.int_]:
        assert map_.shape == self.map_.shape
        state = np.zeros(shape=self.n_elems, dtype=int)
        for (i, j), char in np.ndenumerate(map_):
            ind = self.coords_flattened[i, j]
            state[ind] = int(char == _occupied)
        
        return state
    
    def simulate(self, state: NDArray[np.int_], n_steps=-1) -> NDArray[np.int_]:
        M = self.vicinity_transition_matrix()
        state = state.copy()

        change = True
        nits = 0
        stopat = n_steps if n_steps != 1 else float("inf")

        while change and nits < stopat:
            nits += 1
            print(f"{nits=}")
            next_state = np.zeros(state.shape, dtype=int)
            neighbors_occ = M.dot(state)
            empty = 1 - state

            occ_empty = empty*(neighbors_occ == 0)
            crowded = state*(neighbors_occ >= 4)
            noncrowded = state*(neighbors_occ < 4)

            next_state = (occ_empty + crowded + noncrowded)*self.seat_filter
            assert all(v in (0, 1) for v in next_state)

            change = not np.all(state == next_state)
            
            state = next_state
        
        return state




def solve(data: str) -> tuple[int|str, ...]:
    map_ = parse(data)

    sim = Simulation(map_)

    x0 = sim.ascii_to_state(map_)
    

    x1 = sim.simulate(x0, n_steps=100)
    print(sim.visualize_state(x1))

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
