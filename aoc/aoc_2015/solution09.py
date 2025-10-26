#   `·. · · `  +·.` * ·   ·.`+ ·  ··`       ``  .··* *.  ·•`  · *.    ·•.`·.··  
# `.·.·`+.*•·· `  * ·.*· ` ·. All in a Single Night   `· .  * `·.  · `. +  · *·.
# · .`• .*··.`   ·     https://adventofcode.com/2015/day/9   * `·`   . ·     `.·
# .·   ·`   * .·+` · ·   .  +·`    ·..*  ·`*· .·  *`  ·` ·*  .·. + ``·   · *`·.·


import networkx as nx


def parse(s: str):
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


def solve(data: str) -> tuple[int|str, int|str]:
    G = parse(data)

    paths = grow_all_paths(G)
    shortest: float|int = float('inf')
    longest: float|int = float('-inf')
    for path in paths:
        if len(path) < len(G):
            continue
        length = path_length(path, G)
        shortest = min(shortest, length)
        longest = max(longest, length)
    
    assert isinstance(shortest, int)
    assert isinstance(longest, int)
    star1 = shortest
    print(f"Solution to part 1: {star1}")

    star2 = longest
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 9
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()