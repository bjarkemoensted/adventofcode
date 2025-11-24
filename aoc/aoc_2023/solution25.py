# `· `·.*   · ` ·•.    `·. ·   ·   ·` ··`*  .·  ·*+· . ` · ` ·*.· ·  .`·  · •`·*
# ·•· *  .· `  . ·* ·  + ·.  `·  * Snowverload ·  ·*·+  . ·*. ·` ·*`·   ·.*`·+ ·
# +*· .*··`  · *`.· *· https://adventofcode.com/2023/day/25    ·*   .·   ·`*.·* 
#  . · · `+ . ·    ·`* ·.*·  · .·` .* `· · `·* • ·.·* ` ·.   · •`  *  ··`  ··.` 

import networkx as nx


def parse(s: str) -> nx.Graph:
    G = nx.Graph()
    for line in s.split("\n"):
        u, mess = line.split(": ")
        for v in mess.split():
            G.add_edge(u, v)
    return G


def snip(G, n=3):
    G = G.copy()
    edge_centralities = sorted(nx.edge_betweenness_centrality(G).items(), key=lambda t: -t[1])

    for (u, v), _ in edge_centralities[:n]:
        G.remove_edge(u, v)

    components = nx.connected_components(G)
    return components


def multiply_component_sizes(components):
    res = 1
    for component in components:
        res *= len(component)
    return res


def solve(data: str) -> tuple[int|str, None]:
    G = parse(data)

    parts = snip(G)
    star1 = multiply_component_sizes(parts)
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 25
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
