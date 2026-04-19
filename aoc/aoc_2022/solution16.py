# ·`. ·`  .· ·*    ·.*·.·  • ` ·     · .  *·` .*   ·  `·  .· .  ··   .` ·* ·· `.
# .+·* .`· `• .·  ·*  ·+   ·  Proboscidea Volcanium   .  ·`   ·   `·+  .*·`  ·. 
# ··* ` ·.+ ·. *·.·  · https://adventofcode.com/2022/day/16 .·` ·  *.· `  ·.  *·
# +`··.·. · `•·` ·*     .*·`·.* ·   ··. * .`·. `· `*    ·`* ·+.· `.   ·+` .· .·`

import re
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numba import njit
from numpy.typing import NDArray


@dataclass
class Valve:
    label: str
    flow_rate: int
    neighbors: tuple[str, ...]


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


@njit
def _compute_max_release(
        keys: NDArray[np.int_],
        reliefs: NDArray[np.int_],
        startind=0,
        key=0,
        sum_=0,
        n_workers=1) -> int:
    """Given an array with 'keys' - integers representing the set of opened valves,
    and a coresponding array of 'reliefs', the total pressure released for the optimal way
    of opening the valves, computes the maximum possible relief when using n independent
    workers to traverse the graph and open valves.
    This works by considering all combinations of N paths taken by the N workers, taking
    only the paths with mutually independent subsets of the valves, and computing the
    total relief."""
    
    res = sum_

    for i in range(startind, len(reliefs)):
        if keys[i] & key:
            continue  # Ignore overlapping subsets
        
        if n_workers == 1:
            subres = sum_ + reliefs[i]
        else:
            # If there are more workers, recurse on the remainder of the paths
            subres = _compute_max_release(
                keys=keys,
                reliefs=reliefs,
                startind=i+1,
                key=keys[i]+key,
                sum_=sum_+reliefs[i],
                n_workers=n_workers-1
            )

        if subres > res:
            res = subres
        #
    return res


class Cavern:
    def __init__(self, valves: list[Valve], startat: str="AA") -> None:
        self.startat = startat

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
        
        # Store distances between between relevant valves in a matrix
        self.dists = np.zeros(shape=(self.n_valves, self.n_valves), dtype=int)
        for u, v, d in edges:
            i, j = (self.labels_inverse[node] for node in (u, v))
            self.dists[i, j] = d
        #

    def determine_optimal_paths(
            self,
            pos: int=-1,
            minutes=30,
            valves=0,
            relief=0, d: dict|None=None
            ) -> dict[int, int]:
        """Determines the optimal way of opening every possible subset of valves in the
        specified amount of time.
        Returns a dict mapping each valve subset to the greatest possible amount
        of pressure released."""

        # For initial call, use starting position and empty dict
        if pos == -1:
            pos = self.start_ind
        if d is None:
            d = dict()
        
        # Update result if this path beats the current record for this subset
        improved = relief > d.get(valves, 0)
        if improved:
            d[valves] = relief
        
        # Attempt to move to, and open, each remaining valve
        for target, power in enumerate(self.valve_powers):
            # Skip valve if already opened
            if power & valves:
                continue

            # Ignore valves with zero pressure (should only be the starting valve)
            if self.flow_rates[target] == 0:
                continue
            
            # Time left after opening (need to travel the distance and open)
            remaining = minutes - self.dists[pos, target] - 1
            
            # Ignore if we have no time to get to the valve
            if remaining <= 0:
                continue
            
            # Recurse from the target node
            flow = self.flow_rates[target]*remaining
            self.determine_optimal_paths(
                pos=target,
                minutes=remaining,
                valves=valves+power,
                relief=relief+flow,
                d=d
            )

        return d

    def max_release(self, minutes=30, n_workers=1) -> int:
        """Computes the maximum possible pressure release with the specified
        time and number of workers"""

        # Compute the solution for every valves subset for a single worker
        solutions_single = self.determine_optimal_paths(minutes=minutes)

        # Find the optimal combination of n mutually exclusive solutions
        keys, reliefs = map(np.array, zip(*solutions_single.items()))
        res = _compute_max_release(keys=keys, reliefs=reliefs, n_workers=n_workers)
        return res


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
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
