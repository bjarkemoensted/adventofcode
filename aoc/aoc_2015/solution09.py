import networkx as nx


def parse(s):
    graph = nx.Graph()
    for line in s.split("\n"):
        a, b = line.split(" = ")
        dist = int(b)
        u, v = a.split(' to ')
        graph.add_edge(u, v, weight=dist)

    return graph


def extend_path(path, graph):
    nodes_in_path = set(path)
    last = path[-1]
    res = []
    for node in graph[last]:
        if node in nodes_in_path:
            continue
        newpath = [elem for elem in path] + [node]
        res.append(newpath)
        for extended in extend_path(newpath, graph):
            res.append(extended)

    return res


def grow_all_paths(G):
    paths = []
    for node in G:
        thispath = [node]
        newpaths = extend_path(thispath, G)

        paths += newpaths
    return paths


def path_length(path, graph):
    return sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))


def solve(data: str):
    G = parse(data)

    paths = grow_all_paths(G)
    shortest = float('inf')
    longest = float('-inf')
    for path in paths:
        if len(path) < len(G):
            continue
        length = path_length(path, G)
        shortest = min(shortest, length)
        longest = max(longest, length)

    star1 = shortest
    print(f"Solution to part 1: {star1}")

    star2 = longest
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2015, 9
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
