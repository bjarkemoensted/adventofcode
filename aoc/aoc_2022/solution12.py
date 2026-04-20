# .·+`* ·   `  . `·   ·.`·* +· `    · *· . `• ·   `·   .*·  *·. · `·`•   ·  ·+·.
#  ` * ·.·.*  +`·   `·*  .·  Hill Climbing Algorithm `·   * ·  `*   ·   * ··. `·
# `*·  .`  · ·.*  `· * https://adventofcode.com/2022/day/12  *`  · ` .· ·   *·.+
# · `.·  * .· `  · .+ `·•         .·` ·. ·*  · `.*  `·    · . · `.· *·` +.   · `


import string

import networkx as nx
import numpy as np
from numpy.typing import NDArray

START = "S"
END = "E"


# Map characters to heights
heights = {char: i for i, char in enumerate(string.ascii_lowercase)}
heights[START] = heights["a"]
heights[END] = heights["z"]


def parse(s: str) -> NDArray[np.str_]:
    m = np.array([list(line) for line in s.splitlines()])
    return m


def build_graph(elevation: NDArray[np.int_]) -> nx.DiGraph:
    """Construct a graph where neighboring sites u, v are connected if the height
    difference from u to v is at most 1"""
    G = nx.DiGraph()
    for u, height_u in np.ndenumerate(elevation):
        (i, j) = u
        neighbors = ((i+1, j), (i-1, j), (i, j+1), (i, j-1))
        for v in neighbors:
            if not all(0 <= c < lim for c, lim in zip(v, elevation.shape, strict=True)):
                continue
            ip, jp = v
            height_v = elevation[ip, jp]
            if height_v <= height_u + 1:
                G.add_edge(u, v)
            #
        #

    return G


class HeightMap:
    def __init__(self, M: NDArray[np.str_]) -> None:
        self.M = M.copy()
        self.elevation = np.vectorize(heights.get)(self.M)
        self.G = build_graph(self.elevation)

        self.start_coord = next(coord for coord, char in np.ndenumerate(self.M) if char == START)
        self.target_coord = next(coord for coord, char in np.ndenumerate(self.M) if char == END)
    
    def shortest_trail(self) -> int:
        """Compute the shortest path from the start to the target coordinate"""
        res = nx.shortest_path_length(self.G, source=self.start_coord, target=self.target_coord)
        assert isinstance(res, int)
        return res

    def shortest_path_from_elevation(self, start_level=0, cutoff: int|None=None) -> int:
        """Compute the shortest path from any starting point with the specified elevation,
        to the target. Cutoff can be specified to avoid growing paths beyond known value."""
        start_coords = [
            coord for coord, level in np.ndenumerate(self.elevation)
            if level == start_level
        ]

        lenghts = nx.multi_source_dijkstra_path_length(self.G, start_coords, cutoff=cutoff)
        res = lenghts[self.target_coord]
        assert isinstance(res, int)
        return res
    #


def solve(data: str) -> tuple[int|str, ...]:
    M = parse(data)

    hm = HeightMap(M)
    star1 = hm.shortest_trail()
    print(f"Solution to part 1: {star1}")

    star2 = hm.shortest_path_from_elevation(cutoff=star1)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 12
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
