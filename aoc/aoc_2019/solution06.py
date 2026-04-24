# ·*· .+· ·  `·.•· `  •·.·` · * ·   * ` .·*`    ·  .`·+·* ·`   *  · ` .·   ·`.*·
# *·. `· *.·       · ·`.* · *  Universal Orbit Map `·*    .·` *  ·    · .·` ·  ·
# •.· *  ·  ·*` ·  .+` https://adventofcode.com/2019/day/6 ·*      ·+·*`·. *  ·.
# `• ··  .* +·`   · ·*.·      ·` *·· `•.    ·  ·*`· . ·+ ` . ·  ·*` ·* .` ··`·.+

from heapq import heappop, heappush


def parse(s: str) -> list[tuple[str, str]]:
    res = []
    for line in s.splitlines():
        u, v = line.split(")")
        res.append((u, v))
    return res


class DAG:
    """Directed acyclic graph of orbiting bodies"""

    ROOT = "COM"  # center of mass is the root node

    def __init__(self, *links: tuple[str,str]) -> None:
        self.orbits: dict[str, set[str]] = dict()
        self.orbited_by: dict[str, set[str]] = dict()
        for u, v in links:
            self.add_link(u, v)
    
    def add_link(self, u: str, v: str) -> None:
        """Add information that v orbits u"""
        for node in (u, v):
            self.orbits[node] = self.orbits.get(node, set())
            self.orbited_by[node] = self.orbited_by.get(node, set())
        self.orbits[u].add(v)
        self.orbited_by[v].add(u)

    def count_orbits(self, node: str|None=None, running=0) -> int:
        """Counts the total number of direct and indirect orbits"""
        node = self.ROOT if node is None else node
        res = 0
        running += 1
        
        for v in self.orbits[node]:
            res += running
            res += self.count_orbits(v, running)
        
        return res
    
    def shortest_path(self, source: str, target: str) -> int:
        """Returns the shortest path from the source to the target node"""
        stack = [(0, source)]
        visited = {source}
        while stack:
            dist, u = heappop(stack)
            if u == target:
                return dist
            neighbors = self.orbits[u] | self.orbited_by[u]
            for v in neighbors:
                if v in visited:
                    continue
                visited.add(v)
                heappush(stack, (dist+1, v))

        raise RuntimeError(f"No path from {source} to {target}")


def solve(data: str) -> tuple[int|str, ...]:
    links = parse(data)
    G = DAG(*links)

    star1 = G.count_orbits()
    print(f"Solution to part 1: {star1}")

    distance = G.shortest_path(source="YOU", target="SAN")
    star2 = distance - 2
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 6
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
