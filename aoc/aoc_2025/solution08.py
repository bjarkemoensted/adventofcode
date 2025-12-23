# `·   ·+ • .  ·`*·    `.·*.   `·*·  •· . `.*· · *·     .· `*+· ·.`  ·     · `*·
# ·*·.·  ` .*· ` .  ·  .·   +  · .  Playground  *.`   ··   * ·.` · * `  · . · · 
#  ··`+  .` ·   .· *`· https://adventofcode.com/2025/day/8 .   ·  .··  `+  . ·.`
# ·  · .`*·     · .·   ·`* ·  ` .  ·•.·*  ·*`· .  .`·   +.· ·    *·.    .· ` •.·

from functools import reduce
from operator import mul

import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> NDArray[np.int_]:
    res = np.array([[int(elem) for elem in line.split(",")] for line in s.splitlines()], dtype=int)
    return res


def make_order(dists: NDArray[np.float64]) -> NDArray[np.int_]:
    """Takes an nxn distance matrix. Returns an n x 2 matrix where each
    row corresponds to a i, j-coordinate of the distance matrix,
    if order of the distances, and only where j > i (to avoid
    self-links with distance zero, and avoid double counting due to symmetry)"""
    
    # Filter the coordinates in the upper right corner (j > i)
    dists = dists.copy()
    mask = np.triu(np.ones_like(dists, bool), k=1)
    i_all, j_all = np.where(mask)

    # Sort by their values
    order = np.argsort(dists[i_all, j_all])
    i_sorted = i_all[order]
    j_sorted = j_all[order]
    
    # Turn into an array and return
    m = np.column_stack((i_sorted, j_sorted))
    return m


class Circuitry:
    """Represents the light circuitry and exposes helper methods for iterating through nearest pairs of
    nodes and connecting them"""

    def __init__(self, junctions: NDArray[np.int_]) -> None:
        self.pos = junctions
        self.dists = np.linalg.norm(self.pos[:, None] - self.pos[None, :], axis=-1)
        self.n_nodes, _ = self.dists.shape

        self.order = make_order(self.dists)
        self._order_ind = 0

        # Initially, assign all nodes to their own cluster
        self.assigment = [i for i in range(self.n_nodes)]
        self.clusters: dict[int, set[int]] = {i: {i} for i in range(self.n_nodes)}

    def connect(self, i: int, j: int) -> None:
        """Links nodes i and j. Works by absorbing all nodes from the smaller cluster into the larger one."""
        
        # Order the nodes and their clusters by cluster size
        small, large = sorted((i, j), key=lambda x: len(self.clusters[self.assigment[x]]))
        small_cluster, large_cluster = self.assigment[small], self.assigment[large]
        
        # Update assignments
        for ind in self.clusters[small_cluster]:
            self.assigment[ind] = large_cluster
        
        # Remove small cluster and absorb into the large cluster
        self.clusters[large_cluster] |= self.clusters.pop(small_cluster)

    def next_link(self) -> tuple[int, int, float]:
        """Returns the next closest pair of nodes, and their distance"""
        i, j = map(int, self.order[self._order_ind])
        dist = float(self.dists[i, j])
        self._order_ind += 1
        
        return i, j, dist
    
    def build(self, n_steps_max: int=-1) -> tuple[int, int]:
        """Repeatedly connects the next pair of closes nodes.
        If n_steps_max is provided, connects at most that number of pairs.
        If the circuitry becomes a single giant component (all nodes are part of a single cluster),
        the procedure terminates regardless. The last two nodes connected are returned in a tuple."""

        nits = 0
        while True:
            i, j, _ = self.next_link()
            self.connect(i, j)
            nits += 1
            done = n_steps_max != -1 and nits >= n_steps_max or len(self.clusters) == 1
            if done:
                return i, j
            #
        #
    #


def solve(data: str) -> tuple[int|str, ...]:
    junctions_pos = parse(data)
    builder = Circuitry(junctions_pos)

    # This is just to handle the example data
    n_conns = 10 if len(junctions_pos) < 1000 else 1000
    
    builder.build(n_conns)
    star1 = reduce(mul, sorted(map(len, builder.clusters.values()))[-3:])
    print(f"Solution to part 1: {star1}")

    i_final, j_final = builder.build()
    star2 = junctions_pos[i_final][0]*junctions_pos[j_final][0]
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 8
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
