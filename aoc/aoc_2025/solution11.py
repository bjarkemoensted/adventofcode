# .·· *.·` + ·*`· . ·    ·*·     ·   * ··`.  +   · ` + · .* ·  * · · ` · .+·`·* 
# ·.* `·.·  ·    ·•   ·.+`·    `·+·  Reactor `•·  .*··   *`· ·+.·      .`*·  ··.
# *`.··  .·      `*· · https://adventofcode.com/2025/day/11 *`·   ` .· *·   •.`·
# ·*· .   · * `·  · ·.      · ·*`. · `.  ·+ ·    ` .*`  · · .  · • .· *` ·`· *.·

from collections import Counter
from typing import Iterator


class DAG:
    """A directed acyclic graph (assumed to be acyclic, no checks are made)."""

    def __init__(self) -> None:
        self._nodes: set[str] = set()
        self._edges: dict[str, list[str]] = dict()
    
    def add_node(self, u: str) -> None:
        if u in self._nodes:
            return
        self._nodes.add(u)
        self._edges[u] = []
    
    def add_edge(self, u: str, v: str) -> None:
        for node in (u, v):
            self.add_node(node)
        if v not in self._edges[u]:
            self._edges[u].append(v)
        #
    
    def get_neighbors(self, u: str) -> list[str]:
        """For a node u, returns the possible next nodes"""
        return [v for v in self._edges[u]]

    def iter_paths(self, source: str, *target: str) -> Iterator[tuple[str, int]]:
        """Iterates over paths and their multiplicities.
        Takes a source node, where paths must start, and one or more target nodes, where paths
        will terminate. As the graph is presumed acycled, we only keep track of the current
        head of each path (as it's not possible to revisit nodes anyway), and maintain the number
        of paths leading to each head node.
        Yields tuples of nodes reached and the number of paths going there, e.g.
        ('targt1', n1), ..."""

        # Start with the source node as the current head, multiplicity = 1
        front = Counter([source])
        target_nodes = set(target)
        res = {node: 0 for node in target_nodes}

        while front:
            next_: Counter[str] = Counter()
            for head, n_paths in front.items():
                # If we hit a target, add the number of paths to results
                if head in target_nodes:
                    res[head] += n_paths
                    continue
                # Otherwise, continue with the node's neighbors as the new head(s)
                for v in self.get_neighbors(head):
                    next_[v] += n_paths
                #
            front = next_
        
        yield from res.items()
    #


def parse(s: str) -> DAG:
    G = DAG()
    for line in s.splitlines():
        u, edgepart = line.split(": ")
        for v in edgepart.split():
            G.add_edge(u, v)

    return G


def count_paths(G: DAG, source: str, target: str, must_visit: tuple[str, ...]=()) -> int:
    """Counts the number of paths going from the source to the target node, through
    the must_visit nodes, if specified (if not, all paths from source to target are counted)."""

    # If no constraint on the nodes we must pass through, simply sum all source -> target paths
    if not must_visit:
        return sum(n for _, n in G.iter_paths(source, target))
    
    res = 0

    # Count the paths to each intermediary node
    for node_reached, n_paths in G.iter_paths(source, *must_visit):
        # Recurse on the remaining nodes to be visited
        remaining = tuple(node for node in must_visit if node != node_reached)
        n_further = count_paths(G=G, source=node_reached, target=target, must_visit=remaining)
        res += n_paths*n_further
    
    return res


def solve(data: str) -> tuple[int|str, ...]:
    G = parse(data)

    star1 = count_paths(G=G, source="you", target="out")
    print(f"Solution to part 1: {star1}")

    star2 = count_paths(G=G, source="svr", target="out", must_visit=("dac", "fft"))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 11
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
