# ··` . · *`  ·  .*·•  ` · ·    `  · + ··`.·    ··.`  * · ·    + ·.`*·    .· ·``
# `+.· · . +·. · *     ·  `* .·   A Regular Map . ·+`··.   ·       .·`·  `  ·* ·
# ·`· ·   `·  .·*`  `· https://adventofcode.com/2018/day/20   ·` • · *  ·· `.· .
# *··    ` .`· *· · .` +·   ·*.`·   ·   ` •`·.·*  `.+  · · `.·*       `·  ·.•`··

import networkx as nx
from typing import TypeAlias

coord: TypeAlias = tuple[int, int]
edgetype: TypeAlias = tuple[coord, coord]


dirs = dict(
    W=(0, -1),
    N=(1,  0),
    E=(0,  1),
    S=(-1, 0)
)

def step(pos: coord, dir_: coord, subtract=False, factor: int=1) -> coord:
    """Takes a single step in specified direction (v) from given point (u).
    if subtract, computes u - v instead.
    factor can be used to compute u +/i n*v instead."""
    
    i, j = pos
    fac = factor*(-1 if subtract else +1)
    di, dj = (fac*v for v in dir_)

    res = (i+di, j+dj)
    return res


def find_next(s: str, char: str, startat: int=0):
    """Finds the index of the next occurrence of char in the string, disregarding subpatterns
    in parentheses."""
    
    # Keep track of how many layers deep we are in parentheses
    depth = 0
    incs = {"(": +1, ")": -1}
    
    for ind in range(startat, len(s)):
        # If not inside parentheses and we find the char, we're done
        if depth == 0 and s[ind] == char:
            return ind
        # If we hit a parenthesis, increment/decrement current depth
        depth += incs.get(s[ind], 0)
        assert depth >= 0
    #


def parse(regex: str, head = (0, 0)) -> set[edgetype]:
    """Parses a regex into a set of edges (tuples of connected coordinates)."""
    
    edges: set[edgetype] = set()
    current = head
    i = 0
    
    while i < len(regex):
        char = regex[i]
        match char:
            case "^":
                pass
            case "W" | "N" | "E" | "S":
                # If we hit a direction marker, step to neighbor site
                dir_ = dirs[char]
                adj = step(current, dir_)
                edges.add((current, adj))
                current = adj
            case "(":
                # Find the matching closing parenthesis and recurse on the enclosed substring
                a = i+1
                b = find_next(s=regex, char=")", startat=a)
                edges |= parse(regex[a:b], head=current)
                i = b
            case "|" | "$":
                # Finished an 'or' part of the regex, so reset current node
                current = head
            case _:
                raise RuntimeError(f"Invalid char: {char}")
        
        i += 1
    
    return edges


def build_graph(regex: str) -> nx.Graph:
    """Parse the regex and build a graph with the edges described by it"""
    edges = parse(regex)
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def print_map(G: nx.Graph):
    """Helper method for displaying a graph as an ASCII layout"""
    
    # Find the extreme values for i and j
    limits = tuple((vals[0], vals[-1]) for vals in map(sorted, zip(*G.nodes())))
    (ymin, _), (xmin, _) = limits
    
    # Set up ASCII chars. Each room is separated by a wall, so for n rooms we need 2n+1 chars to have space
    shape = tuple(high - low + 1 for low, high in limits)
    rows, cols = tuple(2*v + 1 for v in shape)
    m = [["#" for _ in range(cols)] for _ in range(rows)]
    
    def center(pos: coord) -> coord:
        """Shifts a coordinate so all coords are >= 0"""
        i, j = pos
        return (i-ymin, j-xmin)
    
    for edge in G.edges():
        # Determine the direction of travel
        u, v = map(center, edge)
        dir_ = step(v, u, subtract=True)
        # Convert to ASCII layout coords and determine the three points (room-door-room) to draw
        ui, uj = u
        draw_coords = [step((2*ui + 1, 2*uj + 1), dir_, factor=n) for n in (0, 1, 2)]
        symbols = ["." for _ in draw_coords]
        symbols[1] = "|" if dir_[0] == 0 else "-"
        for (i, j), symbol in zip(draw_coords, symbols):
            m[i][j] = symbol
    
    # Draw an X at the origin
    i0, j0 = center((0, 0))
    m[i0*2+1][j0*2+1] = "X"        

    lines = ["".join(row) for row in m]
    res = "\n".join(lines)
    print(res)


def solve(data: str) -> tuple[int|str, int|str]:
    # Build graph from the regex and compute all shortest paths from origin
    G = build_graph(data)
    d = nx.single_source_shortest_path_length(G, (0, 0))
    #print_map(G)  # uncomment to show the map as ASCII

    star1 = max(d.values())
    print(f"Solution to part 1: {star1}")

    star2 = sum(v >= 1000 for v in d.values())
    print(f"Solution to part 2: {star2}")
    return star1, star2


def main() -> None:
    year, day = 2018, 20
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
