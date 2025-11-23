# ·  · . .· *`   ·`*·.   ·    ·+`• .`·+·  .  ·* ·`+.  ·  ·`. · *   + ·` ` .· ·*.
# ··`*·  ·.  * · .  `· .· +·` .  Clumsy Crucible · · `  +*·· * ·   `·.  ·*` *.`·
# .•·`*·     ·. `·+    https://adventofcode.com/2023/day/17  .  ·   `  · · *` ·`
# +·.   ·+`··.*  •·`  · * ·   `·..  · ·`+.  `·      · `+·.  ·  `*  .·   +`·. `·+

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
from heapq import heappop, heappush
from typing import Iterator, TypeAlias

import networkx as nx

coordtype: TypeAlias = tuple[int, int]


def parse(s: str) -> dict[coordtype, int]:
    """Map coordinates to their heat losses"""
    d: dict[coordtype, int] = dict()
    for i, row in enumerate(s.splitlines()):
        for j, val in enumerate(row):
            d[(i, j)] = int(val)
        #
    
    return d


@dataclass(frozen=True)
class Crucible:
    direction: coordtype
    position: coordtype
    n_straight: int

    def __lt__(self, other: Crucible) -> bool:
        return self.position > other.position

north = (-1, 0)
west = (0, -1)
east = (0, 1)
south = (1, 0)

_dirchars = {
    north: "^",
    west: "<",
    east: ">",
    south: "v",
    
}

all_dirs = (north, west, south, east)


@cache
def get_turned(dir_: coordtype) -> tuple[coordtype, coordtype]:
    """Returns the two directions which can be obtained by rotating 90 degrees in either direction"""
    i = all_dirs.index(dir_)
    res = tuple(all_dirs[(i + shift) % len(all_dirs)] for shift in (+1, -1))
    assert len(res) == 2
    return res


@cache
def add_coords(a: coordtype, b: coordtype) -> coordtype:
    ia, ja = a
    ib, jb = b
    res = (ia+ib, ja+jb)
    return res


class Grid:
    """Represents the grid where the elves are moving around the lava crucibles.
    This is to collect helper methods for determining heuristics, neighbors, etc in a separate
    class."""
    
    def __init__(self, heat_losses: dict[coordtype, int], target: coordtype=(-1, -1)) -> None:
        self.shape = tuple(max(vals)+1 for vals in zip(*heat_losses.keys()))
        rows, cols = self.shape
        self.target = (rows-1, cols-1) if target == (-1, -1) else target
        self.heat_losses = heat_losses
        
        self.G = nx.DiGraph()
        for u, _ in self.heat_losses.items():
            for v in self.get_neightbors(u):
                weight = self.heat_losses[v]
                self.G.add_edge(u, v, weight=weight)
            #
        
        # Determine the heat loss without any of the imposed constraints. All real heat losses will be >= this value
        G_rev = self.G.reverse(copy=True)
        self._heat_loss_lower_bound = nx.single_source_dijkstra_path_length(
            G=G_rev,
            source=self.target,
            weight="weight"
        )
    
    def heuristic(self, pos: coordtype) -> int:
        """A lower bound on the heat loss when moving a crucible from the input position to the target"""
        return self._heat_loss_lower_bound[pos]
    
    @cache
    def in_bounds(self, pos: coordtype) -> bool:
        return all(0 <= x < lim for x, lim in zip(pos, self.shape, strict=True))
    
    @cache
    def get_neightbors(self, pos: coordtype) -> tuple[coordtype, ...]:
        """Gets the neighbors of the specified site"""
        res_list = []
        for dir_ in all_dirs:
            v = add_coords(pos, dir_)
            if self.in_bounds(v):
                res_list.append(v)
            #
        
        return tuple(res_list)
    
    def as_lists(self) -> list[list[str]]:
        rows, cols = self.shape
        res = [[str(self.heat_losses[(i, j)]) for i in range(rows)] for j in range(cols)]
        return res


class Graph:
    def __init__(self, grid: Grid, n_straight_max: int=3, n_straight_before_turn = 0) -> None:
        self.grid = grid
        self.n_straight_max = n_straight_max
        self.n_straight_before_turn = n_straight_before_turn
    
    def allowed_forward(self, crucible: Crucible) -> bool:
        return crucible.n_straight < self.n_straight_max
    
    def allowed_turn(self, crucible: Crucible) -> bool:
        return crucible.n_straight >= self.n_straight_before_turn
    
    @cache
    def move_forward(self, crucible: Crucible) -> Crucible:
        new_pos = add_coords(crucible.position, crucible.direction)
        res = replace(
            crucible,
            position=new_pos,
            n_straight=crucible.n_straight + 1
        )
        
        return res

    def heuristic(self, crucible: Crucible) -> int:
        res = self.grid._heat_loss_lower_bound[crucible.position]
        return res

    def check(self, crucible: Crucible) -> bool:
        """Whether a crucible's state is valid"""
        if not self.grid.in_bounds(crucible.position):
            return False
        return True
    
    def turn(self, crucible: Crucible, new_dir: coordtype) -> Crucible:
        temp = replace(crucible, direction=new_dir, n_straight=0)
        return self.move_forward(temp)

    def get_neighbors(self, crucible: Crucible) -> Iterator[Crucible]:
        candidates = []
        if self.allowed_forward(crucible):
            candidates.append(self.move_forward(crucible))

        if self.allowed_turn(crucible):
            for new_dir in get_turned(crucible.direction):
                candidates.append(self.turn(crucible, new_dir=new_dir))
            #

        for c in candidates:
            if self.check(c):
                yield c
            #
        #
    
    def total_heat_loss(self, states: list[Crucible]) -> int:
        res = 0
        for state in states[1:]:
            res += self.grid.heat_losses[state.position]
        
        return res
    
    def display_path(self, path: list[Crucible]):
        m = self.grid.as_lists()
        for c in path:
            i, j = c.position
            m[i][j] = _dirchars[c.direction]
        
        s = "\n".join(("".join(line) for line in m))
        print(s)


def a_star(G: Graph, *initial_states: Crucible):
    """Given some initial states, uses A* to determine the optimal path (or one of them) to the target."""
    
    # Variables for keeping track of scores and states
    g0 = 0
    d_g = dict()
    queue: list[tuple[int, Crucible]] = []
    camefrom: dict[Crucible, Crucible] = dict()
    
    # Add the initial states
    for initial_state in initial_states:
        d_g[initial_state] = g0
        f_score = g0 + G.heuristic(initial_state)
        heappush(queue, (f_score, initial_state))
    
    # Keep updating the path with the lowest lower bound, until we hit the target
    while queue:
        _, u = heappop(queue)
        
        # Reconstruct the path when we're finished
        done = u.position == G.grid.target and u.n_straight >= G.n_straight_before_turn
        if done:
            running = u
            path_rev = [u]
            while running in camefrom:
                running = camefrom[running]
                path_rev.append(running)
                
            return path_rev[::-1]
        
        # Check possible subsequent states for any improvement upon the current best path to that state
        for v in G.get_neighbors(u):
            additional_loss = G.grid.heat_losses[v.position]
            g_tentative = d_g[u] + additional_loss
            improved = g_tentative < d_g.get(v, float("inf"))
            
            # If an improvement is found, keep the (new) path to that state
            if improved:
                camefrom[v] = u
                d_g[v] = g_tentative
                h = G.heuristic(v)
                f_score = g_tentative + h
                # Add state to the queue with the (new) lower bound
                heappush(queue, (f_score, v))
            #
        #
    
    raise RuntimeError("No path found")


def solve(data: str) -> tuple[int|str, ...]:
    heat_losses = parse(data)
    grid = Grid(heat_losses)
    
    initial_states = (
        Crucible(position=(0, 0), direction=south, n_straight=0),
        Crucible(position=(0, 0), direction=east, n_straight=0)
    )
    
    G = Graph(grid)
    path1 = a_star(G, *initial_states)
    star1 = G.total_heat_loss(path1)
    print(f"Solution to part 1: {star1}")
    
    G2 = Graph(grid, n_straight_max=10, n_straight_before_turn=4)
    path2 = a_star(G2, *initial_states)
    star2 = G.total_heat_loss(path2)

    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 17
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
