import networkx as nx

# Read in data
with open("input09.txt") as f:
    raw = f.read()


example_input = \
"""London to Dublin = 464
London to Belfast = 518
Dublin to Belfast = 141"""


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


def grow_all_paths(graph):
    paths = []
    for node in G:
        thispath = [node]
        newpaths = extend_path(thispath, graph)

        paths += newpaths
    return paths


def path_length(path, graph):
    return sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))


G = parse(raw)

paths = grow_all_paths(G)
shortest = float('inf')
longest = float('-inf')
for path in paths:
    if len(path) < len(G):
        continue
    length = path_length(path, G)
    shortest = min(shortest, length)
    longest = max(longest, length)


print(f"Shortest path visiting all nodes has length {shortest}.")
print(f"longest path visiting all nodes has length {longest}.")
