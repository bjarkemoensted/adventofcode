# .·`.·   ·` •  ·.·   ` .`· ·.    .·*` ·.   ·`*··   .+*.·`    ·` •    ·*·`` .··*
#  .·· ` .•*·`·  `+.·   ·   .*.  Handy Haversacks ·  . · +.•·     `. ·.  * `·•·.
# ·`  ` ·   `.+     `· https://adventofcode.com/2020/day/7 ·•· .` ·  `*· . ·* .·
# ·+.`*· `·   ·  · `.* ·.  * `·* .·`   `·+.   · .  ·     *· `.*· · ·`   .    ·`.

import networkx as nx
import re


def parse(s: str) -> nx.DiGraph:
    """Constructs a directed graph, with colors as nodes, and edges u -> v denoting
    the number of bags of color v are inside a bag with color u."""

    G = nx.DiGraph()
    pattern = r"(\d+) ([\w\s]+) bag"

    for line in s.splitlines():
        u, children = line.strip().split("bags contain")
        u = u.strip()
        childattrs = []

        for nstring, v in re.findall(pattern=pattern, string=children):
            n = int(nstring)
            G.add_edge(u, v, weight=n)
            childattrs.append((n, v.strip()))
        #

    return G


def count_paths(G: nx.DiGraph, target: str) -> int:
    """Counts the number of nodes which contain a path to the target node."""

    all_colors = set(G.nodes())
    n_paths = 0
    
    for start in (all_colors - set([target])):
        n_paths += nx.has_path(G, start, target)
    
    return n_paths


def count_recursive(G: nx.DiGraph, target: str, recursing=False) -> int:
    """Sums the weights of any path leading to the target."""

    n_paths = 0
    for v in G.neighbors(target):
        contribution = G[target][v]["weight"]*count_recursive(G, v, recursing=True)
        n_paths += contribution
    
    res = n_paths + recursing
    
    return res


def solve(data: str) -> tuple[int|str, ...]:
    G = parse(data)

    goal = "shiny gold"
    star1 = count_paths(G=G, target=goal)
    print(f"Solution to part 1: {star1}")

    star2 = count_recursive(G=G, target=goal)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 7
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
