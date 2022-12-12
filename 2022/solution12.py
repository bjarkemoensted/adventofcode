import networkx as nx
import string


def read_input():
    with open("input12.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def find_elevation(char):
    """Finds elevation from a character in the input (a: 0 etc).
    Uses same elevation for S as a, and for E as z."""

    d = {let: ind for ind, let in enumerate(string.ascii_lowercase)}
    key = char
    if char not in d:
        if char == "S":
            key = "a"
        elif char == "E":
            key = "z"
        #

    res = d[key]
    return res


def allows_travel(source, destination):
    """Determines whether travel from source to destination is possible (elevation can only increment
    by one)"""
    elevation_source = find_elevation(source)
    elevation_destination = find_elevation(destination)
    res = elevation_destination - elevation_source <= 1
    return res


def parse(s):
    """Parses into a bidirectional graph where travel is possible between connected nodes."""
    stuff = s.split("\n")
    G = nx.DiGraph()

    # First add all the nodes and the letters representing their elevation
    nodes = {}
    for i, row in enumerate(stuff):
        for j, u in enumerate(row):
            nodes[(i, j)] = u
            G.add_node((i, j), letter=u)

    for node, letter in nodes.items():
        i, j = node
        for ii, jj in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
            # Drop neighbors that fall off the input array
            if (ii, jj) not in nodes:
                continue

            otherletter = nodes[(ii, jj)]
            # Do not make a connection to nodes we can't travel to
            if not allows_travel(letter, otherletter):
                continue

            G.add_edge((i, j), (ii, jj))

    return G


def find_shortest_path(G, source, target):
    """Returns the length of the shortest path from source to target."""
    try:
        path = nx.shortest_path(G, source=source, target=target)
    except nx.exception.NetworkXNoPath:
        return None

    path_length = len(path) - 1  # path contains the starting node, so length is one step too large
    return path_length


def find_shortest_path_multiple_sources(G, sources, target):
    """Finds the minimum path length among all paths from sources to target."""
    lengths = [find_shortest_path(G, source, target) for source in sources]
    shortest = min([len_ for len_ in lengths if len_ is not None])
    return shortest


def main():
    raw = read_input()
    G = parse(raw)

    source = [u for u in G.nodes if G.nodes[u]["letter"] == "S"][0]
    target = [u for u in G.nodes if G.nodes[u]["letter"] == "E"][0]

    n_steps = find_shortest_path(G, source, target)
    print(f"The shortest path has {n_steps} steps.")

    low_points = [u for u in G.nodes if G.nodes[u]["letter"] in ("S", "a")]
    shortest_hike = find_shortest_path_multiple_sources(G, low_points, target)
    print(f"Shortest path from any lowest point to target is {shortest_hike}.")


if __name__ == '__main__':
    main()
