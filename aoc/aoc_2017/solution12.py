def parse(s):
    """Parse as {node_u: set_of_nodes_connected_with_u}"""
    res = dict()
    for line in s.splitlines():
        a, b = line.split(' <-> ')
        u = int(a)
        neighbors = {int(elem) for elem in b.split(", ")}
        res[u] = neighbors
    return res


def connected_component(connections, starting_node: int):
    """Returns the connected component containing the starting node"""

    component = set([])
    wavefront = {starting_node}
    # Start with the starting node, then repeatedly add neighbors if they're unseen
    while wavefront:
        component = component.union(wavefront)
        wavefront = set.union(*[connections[u] for u in wavefront]) - component

    return component


def get_all_components(connections):
    """Returns a list of all connected components in the graph"""

    missing = set(connections.keys())
    components = []

    while missing:
        seed = list(missing)[0]
        component = connected_component(connections, seed)
        missing -= component
        components.append(component)

    return components


def solve(data: str):
    connections = parse(data)
    component = connected_component(connections, starting_node=0)

    star1 = len(component)
    print(f"Solution to part 1: {star1}")

    components = get_all_components(connections)
    star2 = len(components)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 12
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
