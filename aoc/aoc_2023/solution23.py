# ·*·`.  + · + `     ·* .   .` *·  .`·*    ·  +·  `  *. · `. · + .   ·*  .`•·· ·
#   `·  · .   `+· . ·. · ` * ·.    A Long Walk .`.*·      · .`+  · .· +*`·   .· 
# *· . ·   `.  ·.•·    https://adventofcode.com/2023/day/23   ·   .*  ·.* ·`  ·.
# ·`*    ··.`·.*      ·`·.*`·*   .·`.·  +·.      · `·.*    `   · ` ·    ·*.·.`+·

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import itertools
import numpy as np
from numpy.typing import NDArray
import typing as t


coordtype: t.TypeAlias = tuple[int, int]


class Symbols(str, Enum):
    wall = "#"
    free = "."
    up = "^"
    right = ">"
    down = "v"
    left = "<"


raw = """#.#####################
#.......#########...###
#######.#########.#.###
###.....#.>.>.###.#.###
###v#####.#v#.###.#.###
###.>...#.#.#.....#...#
###v###.#.#.#########.#
###...#.#.#.......#...#
#####.#.#.#######.#.###
#.....#.#.#.......#...#
#.#####.#.#.#########v#
#.#...#...#...###...>.#
#.#.#v#######v###.###v#
#...#.>.#...>.>.#.###.#
#####v#.#.###v#.#.###.#
#.....#...#...#.#.#...#
#.#########.###.#.#.###
#...###...#...#...#.###
###.###.#.###v#####v###
#...#...#.#.>.>.#.>.###
#.###.###.#.###.#.#v###
#.....###...###...#...#
#####################.#"""


up = (-1, 0)
right = (0, 1)
down = (1, 0)
left = (0, -1)

slopes = {
    up: Symbols.up,
    right: Symbols.right,
    down: Symbols.down,
    left: Symbols.left,
}


def parse(s: str) -> NDArray[np.str_]:
    M = np.array([list(line) for line in s.split("\n")])
    return M


@dataclass(frozen=True)
class Section:
    points: tuple[coordtype, ...]
    uphill: bool
    
    @property
    def length(self) -> int:
        return len(self.points)
    #
    
    @property
    def start(self) -> coordtype:
        return self.points[0]
    
    @property
    def end(self) -> coordtype:
        return self.points[-1]
    #


def _determine_junctions(M: NDArray[np.str_]) -> set[coordtype]:
    """Determine the 'junctions' - points where we enter/leave the maze, or where it's possible to turn"""
    junctions: set[coordtype] = set()
    for (i, j), char in np.ndenumerate(M):
        if char == Symbols.wall:
            continue
        
        if i == 0 or i == M.shape[0] - 1:
            junctions.add((i, j))
            continue
        
        n_neighbors = 0
        for di, dj in slopes.keys():
            xp = (i+di, j+dj)
            if all(0 <= c < lim for c, lim in zip(xp, M.shape)):
                n_neighbors += M[*xp] != Symbols.wall
            #
        if n_neighbors > 2:
            assert M[i, j] == Symbols.free
            junctions.add((i, j))
        #
    return junctions


def _iter_sections(M: NDArray[np.str_]) -> t.Iterator[Section]:
    """Generates sections connecting junction points.
    Runs a simple BFS from each junction, growing paths in each possible direction,
    until other junction points are reached. Keeps track of whether the path
    has gone uphill at any slopes"""
    
    junctions = _determine_junctions(M)
    
    for point in sorted(junctions):
        # Initialize paths starting from this junction
        visited: set[coordtype] = {point}
        paths: list[tuple[tuple[coordtype, ...], bool]] = [((point,), False)]  # path so far, uphill
        front: list[tuple[tuple[coordtype, ...], bool]] = []
        
        while paths:
            for path in paths:
                nodes, uphill = path
                i, j = nodes[-1]
                
                # Try moving in all 4 directions
                for dir_ in slopes.keys():
                    di, dj = dir_
                    neighbor = (i+di, j+dj)
                    if not all(0 <= c < lim for c, lim in zip(neighbor, M.shape)):
                        continue  # ignore steps that fall off
                    if M[*neighbor] == Symbols.wall or neighbor in visited:
                        continue  # ignore steps that hit a wall
                    
                    # Update the path nodes and uphill status
                    uphill_here = M[i, j] in slopes.values() and slopes[dir_] != M[i, j]
                    new_nodes = nodes + (neighbor,)
                    new_uphill = uphill or uphill_here
                    
                    # Check if we reached another junction, or need to further grow the path
                    if neighbor in junctions:
                        section = Section(points=new_nodes, uphill=new_uphill)
                        yield section
                    else:
                        visited.add(neighbor)
                        front.append((new_nodes, new_uphill))
                #
            
            paths = front
            front = []
        #
    #


class Graph:
    def __init__(self, M: NDArray[np.str_], allow_uphill=False) -> None:
        self.M = M.copy()
        self.sections = tuple(section for section in _iter_sections(self.M) if allow_uphill or not section.uphill)
        self._section_lookup = {(section.start, section.end): section for section in self.sections}
        
        self._distances: dict[coordtype, dict[coordtype, int]] = {
            node: dict() for section in self.sections for node in (section.start, section.end)
        }
        
        for section in self.sections:
            self._distances[section.start][section.end] = section.length
    
    def nodes(self) -> tuple[coordtype, ...]:
        return tuple(sorted(self._distances.keys()))
    
    def __getitem__(self, u: coordtype) -> dict[coordtype, int]:
        return self._distances[u]
    
    def display_path(self, path: t.Sequence[coordtype]) -> None:
        """Prints a visualization of the input path on the ASCII map, similar to on the AoC web page"""
        M = self.M.copy()
        for i in range(len(path)-1):
            u, v = path[i:i+2]
            section = self._section_lookup[(u, v)]
            for p in section.points:
                M[*p] = "O"
            if i == 0:
                M[*section.points[0]] = "S"
        
        print("\n".join(("".join(line) for line in M)))


def _iter_layers(G: Graph, start: coordtype) -> t.Iterator[tuple[coordtype, ...]]:
    """Iterates over nodes in 'layers', meaning the lowest number of path sections we must traverse
    from start to the node. Iterates tuples of nodes in each layer."""
    visited: set[coordtype] = set()
    front = {start}
    
    while front:
        yield tuple(sorted(front))
        visited |= front
        front = {neighbor for node in front for neighbor in G[node] if neighbor not in visited}
    #


def _collapse_links(*links: tuple[coordtype, coordtype]) -> list[tuple[coordtype, coordtype]]:
    """Takes a number of edges and 'collapses' them by collecting into the outermost points on each line segment.
    For example, a line a -> b -> -> c -> ... -> z is collapsed into (a, z)."""
    
    remaining = {u: v for u, v in links}
    assert len(remaining) == len(links)
    
    res = []
    
    while remaining:
        # Start with a random node, follow the links until we hit a dead end
        u = next((k for k in remaining.keys()))
        v = u
        while v in remaining:
            v = remaining.pop(v)
        
        # Found the two nodes where this segment points to the next layer. Add to results.
        assert u != v
        res.append((u, v))
    
    return res


def longest_path(G: Graph, start: coordtype, end: coordtype) -> list[coordtype]:
    
    layers = list(_iter_layers(G, start))
    assert len(layers[-1]) == 1 and layers[-1][0] == end
    
    # Represent each candidate path by its 'signature' tuple of tuples of destination nodes in edges to next layer,
    # grouped by connected line segments. For example, ((p1,), (p2, p3,)) means one segment points 'down' to the next
    # layer (to node p1), and another segment has 2 ends the point to the next layer at p2 and p3.
    sigtype: t.TypeAlias = tuple[tuple[coordtype, ...], ...]
    signature2best_length: dict[sigtype, int] = dict()
    
    
    for i in range(len(layers)-1):
        this_layer, next_layer = layers[i], layers[i+1]
        dests = set(this_layer) | set(next_layer)
        valid_edges = [(u, v) for u in this_layer for v in G[u] if v in dests]
        combs = itertools.chain(*(itertools.combinations(valid_edges, n) for n in range(1, len(valid_edges)+1)))
        
        print(f"Layer: {i} - {this_layer}")
        
        for comb in combs:
            # An odd number of path segments must pass through each layer, or we won't reach the final layer
            links_forward = [(u, v) for u, v in comb if v in next_layer]
            links_within_layer = [(u, v) for u, v in comb if v in this_layer]
            
            if len(links_forward) % 2 == 0:
                continue
            
            # Count number of connections to/from each node. None must be > 1
            from_, to_ = zip(*comb)
            if any(len(links) != len(set(links)) for links in (from_, to_)):
                continue
            
            internal_endpoints = _collapse_links(*links_within_layer)
            for ep in internal_endpoints:
                print(f"{ep=}", links_forward)
            
            if i == 0:
                assert len(to_) == 1 and len(comb) == 1
                edges_used = comb
                signature = (to_,)
                print(f"OMGOMG {to_=}, {from_=}, {comb=}")
                print(f"SIG: {signature=}")
            
            #G.display_path(comb[0])
            
        
        if i >= 1:
            break  # !!!

    raise NotImplementedError
    

def solve(data: str) -> tuple[int|str, ...]:
    M = parse(data)
    
    G = Graph(M, allow_uphill=True)
    
    start = min(G.nodes())
    end = max(G.nodes())

    
    path1 = longest_path(G, start, end)
    # TODO solve puzzle
    star1 = None
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 23
    from aocd import get_data
    #raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
