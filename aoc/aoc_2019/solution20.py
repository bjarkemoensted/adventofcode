# `. ·*` ·+     ·  `· *` ·*. .+·  ` *       · *·+`     *·.`·.  ·  *`   ·.•`·+ .*
# *·`. ·   ·    + .  *`·  . `·*`+`* Donut Maze *   ·  `. · +*     ·.`· *   `·+`·
# .*·  *`  .· *   ·  · https://adventofcode.com/2019/day/20      ·  .* .·  + ·`.
# ·  +`.* ·`* ·      .+ ` ··+ ` · ·*   .` ·    `  ·*.`*· *+ · .  + ·· `   + * ·`

from collections import defaultdict, deque
from enum import StrEnum
from typing import Iterator, NamedTuple

import numpy as np
from numpy.typing import NDArray

type coord = tuple[int, int]


class Symbol(StrEnum):
    SPACE = "."
    EMPTY = " "
    WALL = "#"


dirs = ((+1, 0), (0, +1), (-1, 0), (0, -1))


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line) for line in s.splitlines()])
    return res


def add_tuples(*x: coord) -> coord:
    i = 0
    j = 0
    for di, dj in x:
        i += di
        j += dj
    
    return i, j


def build_graph(m: NDArray[np.str_]) -> dict[coord, list[coord]]:
    nodes = ((i, j) for (i, j), char in np.ndenumerate(m) if char == Symbol.SPACE)
    G: dict[coord, list[coord]] = {(i, j): [] for i, j in nodes}

    for u in sorted(G.keys()):
        for delta in dirs:
            # Connect neighbors
            v = add_tuples(u, delta)
            if v in G:
                G[u].append(v)
            #
        #
    return G        


def iterate_portals(m: NDArray[np.str_]) -> Iterator[tuple[coord, str, bool]]:
    """Discovers the portals in an ASCII map.
    Iterates over tuples of
    1) the coordinate adjacent to the portal,
    2) the label of the portal
    3) a bool indicating whether the portal recurses into a smaller maze or out"""
    
    for u, char in np.ndenumerate(m):
        if char != Symbol.SPACE:
            continue

        label_starts = [(dir_, add_tuples(u, dir_)) for dir_ in dirs]
        for dir_, p in label_starts:
            label = ""
            recurse: bool|None = None
            char = m[*p].item()
            while char.isupper():
                label += char
                p = add_tuples(p, dir_)
                falloff = not all(0 <= c < lim for c, lim in zip(p, m.shape, strict=True))
                if falloff:
                    recurse = False
                    break
                char = m[*p]
                if char == Symbol.EMPTY:
                    recurse = True
                #
            if label:
                assert recurse is not None
                flip = any(elem < 0 for elem in dir_)
                if flip:
                    label = label[::-1]
                yield u, label, recurse
            #
        #
    #


class State(NamedTuple):
    pos: coord
    level: int=0


class Maze:
    START_LABEL = "AA"
    END_LABEL = "ZZ"

    def __init__(self, arr: NDArray[np.str_]) -> None:
        self.arr = arr.copy()
        self.portals: dict[coord, tuple[coord, int]] = dict()

        self.start = (-1, -1)
        self.end = (-1, -1)
        
        self.G = build_graph(self.arr)
        temp = defaultdict(list)
        for p, label, recurse in iterate_portals(self.arr):
            if label == self.START_LABEL:
                self.start = p
            elif label == self.END_LABEL:
                self.end = p
            shift = +1 if recurse else -1
            temp[label].append((p, shift))
        
        for label, elems in temp.items():
            assert len(elems) == 2 or len(elems) == 1 and label in (self.START_LABEL, self.END_LABEL)
            for i, (source, shift) in enumerate(elems):
                dest, _ = elems[(i+1) % len(elems)]
                assert source not in self.portals
                self.portals[source] = (dest, shift)
            #

    def get_neighbors(self, state: State, recursive: bool) -> Iterator[State]:
        """Get the neighbors to the input state.
        If recursive, allows to portal in and out of sub-mazes."""

        # Move to immediate neighbors
        for adj in self.G[state.pos]:
            yield State(pos=adj, level=state.level)
        
        # Nothing to do if current position isn't a portal
        if state.pos not in self.portals:
            return
        
        # Don't use the starting position as portal
        if state.pos == self.start:
            return
        
        # Only use the end portal if we're at the initial level
        if state.pos == self.end and state.level != 0:
            return
        
        # If no recursion, we don't shift levels except to -1 when exiting the maze
        dest, shift = self.portals[state.pos]
        if not recursive and state.pos != self.end:
            shift = 0
        
        # Do the teleport, unless we portal to a negative level
        new_level = state.level+shift
        if new_level >= 0 or state.pos == self.end:
            yield State(pos=dest, level=state.level+shift)


    def shortest_path(self, recursive=False) -> int:
        """Detemine shortest path to the target"""
        s0 = State(self.start)
        
        target = State(pos=self.end, level=-1)

        visited = {s0}
        queue: deque[tuple[State, int]] = deque()
        queue.append((s0, 0))

        while queue:
            u, dist = queue.popleft()

            for v in self.get_neighbors(state=u, recursive=recursive):
                if v in visited:
                    continue
                visited.add(v)

                if v == target:
                    return dist
                
                queue.append((v, dist+1))

        raise RuntimeError




def solve(data: str) -> tuple[int|str, ...]:
    arr = parse(data)
    maze = Maze(arr)

    star1 = maze.shortest_path()
    print(f"Solution to part 1: {star1}")

    star2 = maze.shortest_path(recursive=True)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 20
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
