# .`·`.* ·`      `· . *   ·· *` . ·  +`  · .`• · ` · * ` ·    ` .   .`··* +·.  `
# *·.  .·`·+  `  . ·`  *.    · `· Blizzard Basin ·`  `. ` ·*  .•· `  ·     ` .· 
# `  .*     ·`  •·`.   https://adventofcode.com/2022/day/24    ·`  .· * ·+`  `.·
# ·.+` ·   .`*·`  . +  ·`·     ·*`    ·. ·*`         ·*  .``     ` · * . ·+.· ` 

from collections import defaultdict
from functools import cache
from heapq import heappop, heappush

import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple

raw = """#.######
#>>.<^<#
#.<..<<#
#>v.><>#
#<^v^^>#
######.#"""

SPACE = "."
WALL = "#"
EXPEDITION = "E"

dtype = np.int64
type coordtype = tuple[int, int]

directions: dict[str, coordtype] = {
    ">": (0, 1),
    "<": (0, -1),
    "v": (1, 0),
    "^": (-1, 0),
}

directions_inv = {v: k for k, v in directions.items()}


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line) for line in s.splitlines()])
    return res


@cache
def step(pos: coordtype, dx: coordtype, limits: tuple[coordtype, coordtype]) -> coordtype:
    """Returns the coordinates one step in the direction dx, wrapping around if exceeding limits."""

    res_list = []
    for p, delta, (low, high) in zip(pos, dx, limits):
        pp = p + delta
        if pp < low:
            pp = high - 1
        elif pp >= high:
            pp = low
        res_list.append(pp)
    
    a, b = res_list
    return a, b


class State(NamedTuple):
    """Represents a state on the elves' journey to the extraction point.
    pos: Coordinates (i, j).
    tick: The time, used to lookup blizzard positions"""

    pos: coordtype
    tick: int
    destinations: tuple[coordtype, ...]


def _cache_blizzard_movements(
        blizzards: dict[coordtype, coordtype],
        limits: tuple[coordtype, coordtype]
    ) -> list[list[tuple[coordtype, coordtype]]]:
    """Takes a dictionary mapping coordinates to blizzard valocities at that coordinate.
    The blizzards moving horizontally and vertically will have different periods, so this function computes
    2 lists, each storing the locations of blizzards moving in one plane, at each time point before recurrence.
    """

    res = [list(blizzards.items())]

    while True:
        next_state = [(step(pos, dx, limits), dx) for pos, dx in res[-1]]
        if next_state == res[0]:
            break
        res.append(next_state)

    return res


@cache
def manhatten_dist(c1: coordtype, c2: coordtype) -> int:
    res = sum(abs(b - a) for a, b in zip(c1, c2, strict=True))
    return res


class Graph:
    def __init__(self, map_: NDArray[np.str_]) -> None:
        self.M = map_.copy()
        nr, nc = self.M.shape
        self.shape = nr, nc
        # Limits, excluding the walls
        self.limits = ((1, nr - 1), (1, nc - 1))

        blizzard_pos: dict[coordtype, coordtype] = dict()
        for (i, j), char in np.ndenumerate(self.M):
            if char in directions:
                blizzard_pos[(i, j)] = directions[char]
                self.M[i, j] = SPACE
            #

        self._blizzard_states_full = _cache_blizzard_movements(blizzard_pos, limits=self.limits)
        self.blizzard_ticks = tuple({pos for pos, _ in bt} for bt in self._blizzard_states_full)

        free_inds = [(i, j) for (i, j), char in np.ndenumerate(self.M) if char == SPACE]
        self.start = free_inds[0]
        self.target = free_inds[-1]
        self.free_inds = set(free_inds)

    @cache
    def _get_next_positions(self, i: int, j: int) -> list[coordtype]:
        """Determines the next possible positions, given a current position."""
        
        res = []
        steps = [(0, 0)] + list(directions.values())
        for di, dj in steps:
            p = (i+di, j+dj)
            if p in self.free_inds:
                res.append(p)
        
        return res

    def get_neighbors(self, state: State) -> list[State]:
        tick_post = (state.tick + 1) % len(self.blizzard_ticks)
        # Keep the possible next state where the expedition does not end up in a blizzard
        blizz = self.blizzard_ticks[tick_post]
        next_pos = (pos for pos in self._get_next_positions(*state.pos) if pos not in blizz)
        res = []
        for pos in next_pos:
            dests = state.destinations[1:] if state.destinations[0] == pos else state.destinations
            neighbor = State(pos=pos, tick=tick_post, destinations=dests)
            res.append(neighbor)

        return res

    def lower_bound(self, state: State) -> int:
        res = 0
        p = state.pos
        for point in state.destinations:
            res += manhatten_dist(p, point)
            p = point
        return res

    def get_initial_state(self, there_and_snack_again=False) -> State:
        """Get the starting state for the travels.
        If there_and_snack_again is True, the points target -> start -> target will be
        queued in the state. Otherwise, only the target will be used."""
        
        destinations = (self.target, self.start, self.target) if there_and_snack_again else (self.target,)
        res = State(pos=self.start, tick=0, destinations=destinations)
        return res

    def display(self, state: State) -> None:
        M = self.M.copy()
        blizz = defaultdict(list)
        for pos, dx in self._blizzard_states_full[state.tick]:
            blizz[pos].append(directions_inv[dx])
        
        for (i, j), elems in blizz.items():
            M[i, j] = elems[0] if len(elems) == 1 else str(len(elems))
        
        M[*state.pos] = EXPEDITION
        print("\n".join(["".join(row) for row in M]))
    #


class PriorityQueue[T]:
    """Simple priority queue for the A* implementation. This is faster than putting State objects directly on a list
    with heapq, because adding a tiebreaker integer means we don't need comparison operations on state instances."""

    def __init__(self) -> None:
        self.a: list[tuple[int, int, T]] = []
        self.counter = 0
    
    def push(self, priority: int, elem: T) -> None:
        heappush(self.a, (priority, self.counter, elem))
        self.counter += 1
    
    def pop(self) -> T:
        _, _, res = heappop(self.a)
        return res


def a_star(G: Graph, initial_state: State, target: coordtype) -> list[State]:
    """Uses A* to find the shortest path to the target state."""
    
    f0 = G.lower_bound(initial_state)
    queue: PriorityQueue[State] = PriorityQueue()
    queue.push(f0, initial_state)
    d_g = {initial_state: 0}
    camefrom: dict[State, State] = dict()

    while queue:
        u = queue.pop()

        done = u.pos == target and not u.destinations
        if done:
            node = u
            path = [node]

            while node in camefrom:
                node = camefrom[node]
                path.append(node)
            path.reverse()
            return path
        
        for v in G.get_neighbors(u):
            delta = 1

            g_tentative = d_g[u] + delta
            improved = g_tentative < d_g.get(v, float("inf"))
            if improved:
                d_g[v] = g_tentative
                h = G.lower_bound(v)
                f_score = g_tentative + h
                camefrom[v] = u
                queue.push(f_score, v)

    raise RuntimeError("No path found")


def solve(data: str) -> tuple[int|str, ...]:
    M = parse(data)
    
    G = Graph(M)
    initial_state = G.get_initial_state()

    path = a_star(G, initial_state, target=G.target)
    star1 = len(path) - 1
    print(f"Solution to part 1: {star1}")

    initial_state_2 = G.get_initial_state(there_and_snack_again=True)
    path = a_star(G, initial_state_2, target=G.target)
    star2 = len(path) - 1

    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 24
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
