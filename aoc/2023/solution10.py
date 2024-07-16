from functools import cache
import networkx as nx
import numpy as np


def read_input():
    with open("input10.txt") as f:
        puzzle_input = f.read()

    return puzzle_input


def parse(s):
    chars = [list(line) for line in s.split("\n")]
    M = np.array(chars)
    return M


north = np.array([-1, 0])
south = np.array([1, 0])
west = np.array([0, -1])
east = np.array([0, 1])
_all_dirs = (north, south, east, west)


_char2directions = {
    "|": (north, south),
    "-": (east, west),
    "L": (north, east),
    "J": (north, west),
    "7": (south, west),
    "F": (south, east)
}


@cache
def rotate(dir_):
    """Takes a direction, e.g. (1, 0) for due south, and returns the direction win which the sites to the immediate
    left and right, respectively, lie when facing that direction. For example west, east when direction is south."""
    directions = [tuple(vec) for vec in [north, east, south, west]]
    for i, direction in enumerate(directions):
        if direction == dir_:
            left = directions[i-1]
            right = directions[(i+1) % len(directions)]
            return left, right
    raise ValueError


def trace_adjacent_sites_to_cycle(cycle):
    """Takes a cycle consisting of a list of pairs of nodes, each of which being a coordinate, e.g. (42, 60).
    Iterates over pairs of nodes lying to the left and right of each node when walking along the cycle.
    A pair of coordinates is yielded before and after updating the direction, so duplicates occur when direction
    does not change."""

    for u, v in cycle:
        dir_ = tuple(c2 - c1 for c1, c2 in zip(u, v))
        for node in (u, v):
            adj = []
            for ndir in rotate(dir_):
                adjacent_site = tuple(c1 + c2 for c1, c2 in zip(node, ndir))
                adj.append(adjacent_site)
            yield tuple(adj)


def get_neighbors(M, i, j, *directions):
    """Takes a numpy array M and row + col indices i, j, and an iterable of directions, e.g. (-1, 0).
    Returns a list of the sites lying in each of those directions from i, j, excluding sites out of bounds."""
    coord = np.array([i, j])
    res = []
    for dir_ in directions:
        newcoord = coord + dir_
        withinbounds = all(0 <= val < lim for val, lim in zip(newcoord, M.shape))
        if not withinbounds:
            continue
        res.append(tuple(newcoord))

    return res


def build_graph_from_sketch(M):
    """Takes an array of characters and returns the graph it represents.
    The true symbol which the S-symbol is covering is inferred by trying all possible values, keeping the graph
    in which the loop containing S is the longest possible."""

    nrows, ncols = M.shape
    ij2neighbors = dict()
    S_coords = None

    for i in range(nrows):
        for j in range(ncols):
            char = M[i, j]
            if char == "S":
                S_coords = (i, j)
                ij2neighbors[(i, j)] = []
            if char not in _char2directions:
                continue

            # Note down the neighboring sites which this site points to
            directions = _char2directions[char]
            ij2neighbors[(i, j)] = get_neighbors(M, i, j, *directions)
        #

    # Construct graphs for all possible values of S
    longest_loop = float("-inf")
    G_best = None
    for S_char, directions in _char2directions.items():
        i_s, j_s = S_coords
        s_neighbors = get_neighbors(M, i_s, j_s, *directions)
        # If this value of S doesn't connect to neighboring sites, there will not be a loop and the graph is invalid
        if not all(neighbor in ij2neighbors for neighbor in s_neighbors):
            continue

        ij2neighbors[S_coords] = s_neighbors
        G = nx.Graph()

        # Connect all sites that point to each other
        for site, neighbors in ij2neighbors.items():
            for neighbor in neighbors:
                if site in ij2neighbors.get(neighbor, []):
                    G.add_edge(site, neighbor)
                #
            #
        # Check length of the cycle containing S. Keep graph if it beats the record so far
        try:
            cycle_len = len(nx.find_cycle(G, source=S_coords))
        except nx.exception.NetworkXNoCycle:
            continue
        if cycle_len > longest_loop:
            G_best = G
            longest_loop = cycle_len

    return S_coords, G_best


def find_distance(G, coords):
    """Finds the max distance from S along G. Since S is in a loop, max dist is half the cycle length."""
    cycle = nx.find_cycle(G, source=coords)
    res = len(cycle) // 2
    return res


def get_blop(M, i, j, forbidden_sites):
    """Takes a char array, coords i, j, and a list of 'forbidden' coords (those occupied by the cycle).
    Returns a list of the sites directly connected to i, j by iteratively adding the neighbors to nodes in the set
    starting from i, j."""

    res = set([])
    coords = [(i, j)]

    while coords:
        new_coords = []
        for coord in coords:
            if coord not in forbidden_sites:
                # If sites is not part of the cycle, add it to this blop
                a, b = coord
                res.add(coord)
                # Consider all the nodes' neighbors (except those already seen) in the next iteration
                for ncoord in get_neighbors(M, a, b, *_all_dirs):
                    if ncoord not in res:
                        new_coords.append(ncoord)
                #
            #
        coords = sorted(set(new_coords))

    return res


def _map_blop_to_nodes(node2blop):
    """Takes a mapping from node to blop numbers, e.g.
    {(42, 60): 0, (43, 60): 0}, (80, 13): 1}.
    Returns a corresponding mapping from blop numbers to the set of nodes in that blop, e.g.
    {0: {(42, 60), (43, 60)}, 1: {(80, 13)}}"""

    blop2nodes = dict()
    for node, ind in node2blop.items():
        try:
            blop2nodes[ind].add(node)
        except KeyError:
            blop2nodes[ind] = set([node])
        #
    return blop2nodes


def get_blops(M, forbidden_sites):
    """Takes a char array M and list of forbidden nodes. Identifies 'blops' in the nodes which are in direct contact
    with each other. Returns a mapping from each node to the number of the blop in which it is contained."""
    nrows, ncols = M.shape
    blop_number = 0
    node2blop = dict()  # node to blop number
    for i in range(nrows):
        for j in range(ncols):
            coord = (i, j)
            if coord in node2blop:
                continue
            new_blop = get_blop(M, i, j, forbidden_sites)
            if new_blop:
                for blopcoord in new_blop:
                    node2blop[blopcoord] = blop_number
                blop_number += 1
            #
        #
    return node2blop


def _node_touches_edge(node, M):
    return any(c == 0 or c == lim - 1 for c, lim in zip(node, M.shape))


def count_enclosed_tiles(G, s_coord, M):
    """Counts the number of tiles enclosed by the cycle in G containing the starting node s_coord."""

    # Identify the cycle in which S resides, and the blops of sites which touch each other.
    cycle = nx.find_cycle(G, source=s_coord)
    nodes_in_cycle = [u for u, _ in cycle]
    forbidden_sites = set(nodes_in_cycle)
    node2blop = get_blops(M, forbidden_sites)
    blop2nodes = _map_blop_to_nodes(node2blop)

    # When nodes touch the edge, all nodes in that blop are not enclosed by the cycle
    blops_outside = {blop for blop, nodes in blop2nodes.items() if any(_node_touches_edge(node, M) for node in nodes)}

    # Get collection of nodes which lie on the left and right sides of the cycle
    touch_points = trace_adjacent_sites_to_cycle(cycle)
    left, right = zip(*touch_points)
    for nodes in (left, right):
        # This is the outer edge of we touch a) the map edge or b) nodes that are not enclosed
        nodes_not_in_cycle = [node for node in nodes if node not in forbidden_sites]
        outer_edge = False
        for node in nodes:
            if node in forbidden_sites:
                continue
            if node not in node2blop:
                outer_edge = True
                continue
            elif node2blop[node] in blops_outside:
                outer_edge = True

        # If outer edge, nodes in the blops touched are also not enclosed
        blops_touched = {node2blop[node] for node in nodes_not_in_cycle if node in node2blop}
        if outer_edge:
            blops_outside.update(blops_touched)

    # All tiles not enclosed are enclosed in the cycle
    n_inside = sum(len(nodes) for blop, nodes in blop2nodes.items() if blop not in blops_outside)
    return n_inside


def main():
    raw = read_input()
    M = parse(raw)

    S_coords, G = build_graph_from_sketch(M)
    star1 = find_distance(G, S_coords)
    print(f"The point furthest from S is {star1} steps away.")

    star2 = count_enclosed_tiles(G, S_coords, M)
    print(f"Tiles enclosed by the loop: {star2}.")


if __name__ == '__main__':
    main()
