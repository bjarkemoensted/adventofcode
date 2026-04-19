# ·`. ·`  .· ·*    ·.*·.·  • ` ·     · .  *·` .*   ·  `·  .· .  ··   .` ·* ·· `.
# .+·* .`· `• .·  ·*  ·+   ·  Proboscidea Volcanium   .  ·`   ·   `·+  .*·`  ·. 
# ··* ` ·.+ ·. *·.·  · https://adventofcode.com/2022/day/16 .·` ·  *.· `  ·.  *·
# +`··.·. · `•·` ·*     .*·`·.* ·   ··. * .`·. `· `*    ·`* ·+.· `.   ·+` .· .·`

from dataclasses import dataclass
from itertools import combinations
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import re
from typing import Iterator


raw = """Valve AA has flow rate=0; tunnels lead to valves DD, II, BB
Valve BB has flow rate=13; tunnels lead to valves CC, AA
Valve CC has flow rate=2; tunnels lead to valves DD, BB
Valve DD has flow rate=20; tunnels lead to valves CC, AA, EE
Valve EE has flow rate=3; tunnels lead to valves FF, DD
Valve FF has flow rate=0; tunnels lead to valves EE, GG
Valve GG has flow rate=0; tunnels lead to valves FF, HH
Valve HH has flow rate=22; tunnel leads to valve GG
Valve II has flow rate=0; tunnels lead to valves AA, JJ
Valve JJ has flow rate=21; tunnel leads to valve II"""


@dataclass
class Valve:
    label: str
    flow_rate: int
    neighbors: tuple[str, ...]


def compute_pairwise_dists(*valves: Valve) -> dict[str, dict[str, int]]:
    """Returns a graph connecting all valves"""

    # Build the graph
    G = nx.Graph()
    for valve in valves:
        u = valve.label
        for v in valve.neighbors:
            G.add_edge(u, v)
        #
    
    # Compute all distances between valves
    res: dict[str, dict[str, int]] = dict()
    for u, d in nx.all_pairs_dijkstra_path_length(G):
        res[u] = dict()
        for v, dist in d.items():
            assert isinstance(dist, int)
            res[u][v] = dist
        
    return res


def parse(s: str) -> list[Valve]:
    res = []
    for line in s.split("\n"):
        match = re.match(r'Valve (.*) has flow rate=(.*);(.*)', line)
        assert match is not None
        label, flow_str, tunnels_str = match.groups()
        flow_rate = int(flow_str)
        tunnels = tuple(s_.replace(",", "") for s_ in tunnels_str.split()[4:])
        
        valve = Valve(label=label, flow_rate=flow_rate, neighbors=tunnels)
        res.append(valve)

    return res


class Cavern:
    def __init__(self, valves: list[Valve], elephant=False, startat: str="AA", total_time=30) -> None:
        self.startat = startat
        self.minutes_total = total_time
        self.n_workers = 2 if elephant else 1

        # Keep start position and any nonzero-flow valve, ordered by flows
        valves_ordered = sorted(
            (v for v in valves if v.flow_rate > 0 or v.label == self.startat),
            key=lambda valve: valve.flow_rate,
            reverse=True
        )

        # setup arrays of flow rates, indices, and number of valves (excluding non-zero non-start)
        self.flow_rates = np.array([v.flow_rate for v in valves_ordered])
        
        self.valve_inds = np.arange(len(self.flow_rates))
        self.n_valves = len(self.valve_inds)
        self.valve_powers = np.array([1 << i for i in range(self.n_valves)])
        self.max_flow = sum(self.flow_rates)  # Flow when all valves are open

        # Set the labels, and make a mapping from labels to indices
        self.valve_labels = tuple(v.label for v in valves_ordered)
        self.labels_inverse = {label: i for i, label in enumerate(self.valve_labels)}
        self.start_ind: int = self.labels_inverse[self.startat]
        
        # Compute all pairwise distances
        pw = compute_pairwise_dists(*valves)
        edges = (
            (u, v, dist)
            for u, d in pw.items() for v, dist in d.items()
            if all(node in self.labels_inverse for node in (u, v))
        )
        
        # Store distances between valves in a matrix
        self.dists = np.zeros(shape=(self.n_valves, self.n_valves), dtype=int)
        for u, v, d in edges:
            i, j = (self.labels_inverse[node] for node in (u, v))
            self.dists[i, j] = d
        #

    def bfs(self, pos: int=-1, minutes=-1, valves=0, relief=0, d: dict|None=None) -> dict[int, int]:
        if pos == -1:
            pos = self.start_ind
        if minutes == -1:
            minutes = self.minutes_total
        if d is None:
            d = dict()
        
        improved = relief > d.get(valves, 0)
        if improved:
            d[valves] = relief
        
        for target, power in enumerate(self.valve_powers):
            remaining = minutes - self.dists[pos, target] - 1
            if remaining <= 0 or power & valves:
                continue
            
            flow = self.flow_rates[target]*remaining
            if flow == 0:
                continue
            self.bfs(pos=target, minutes=remaining, valves=valves+power, relief=relief+flow, d=d)

        return d

    def max_release(self, minutes=30, n_workers=1) -> int:
        d = {int(k): int(v) for k, v in self.bfs(minutes=minutes).items()}

        best = -1

        keys, reliefs = map(np.array, zip(*d.items()))
        print(reliefs)

        a, b = zip(*d.items())
        
        hmm = combinations(d.items(), n_workers)
        for elem in hmm:
            valve_sets, reliefs = zip(*elem)
            
            overlap = False
            running = 0
            for vs in valve_sets:
                if vs & running:
                    overlap = True
                    break
                else:
                    running += vs
                #
            
            if not overlap:
                best = max(best, sum(reliefs))
            
        return best
    #


def solve(data: str) -> tuple[int|str, ...]:
    valves = parse(data)
    cavern = Cavern(valves)

    star1 = cavern.max_release(minutes=30, n_workers=1)
    print(f"Solution to part 1: {star1}")

    star2 = cavern.max_release(minutes=26, n_workers=2)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 16
    from aocd import get_data
    #raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
