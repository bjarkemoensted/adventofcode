#  * ` ꞏ    ` ⸳*ꞏ.   `*     ` .*+ꞏ`        ⸳ꞏ  `.•  ⸳     +⸳`  .⸳` • ꞏ⸳ `+ .`* +
# ꞏ`+.*    ⸳ . ꞏ * ` ꞏ`.  + * ꞏ   .  Hoof It `ꞏ   . .`⸳ ꞏ*ꞏ+    `. +  `*    ꞏꞏ ⸳
#   `+⸳ `  `.   ⸳⸳•ꞏ`  https://adventofcode.com/2024/day/10   ⸳ * `.*`    `ꞏ• .`
# +.⸳   ꞏ`. •`   ꞏ⸳ *.  ⸳`  •ꞏ   ⸳` ꞏ ꞏ*`.⸳+`  .ꞏ    ꞏ.  `   ꞏ   ⸳ `⸳++ꞏ   `.`* 


from collections import defaultdict
import numpy as np


def parse(s):
    m = np.array([[int(x) for x in line] for line in s.splitlines()])
    return m


class Graph:
    """Graph for representing the topographic map"""

    dirs = (
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0)
    )
    
    def __init__(self, m: np.ndarray):
        # Keep track of edges, and the indices of each value encountered
        self.edges = dict()
        self.value_lookup = defaultdict(lambda: [])
        
        # Go over all indices and values in the map
        for u in np.ndindex(m.shape):
            val = m[u]
            self.value_lookup[val].append(u)
            self.edges[u] = []
            
            # Go over all neighbors (ESNW)
            for delta in self.dirs:
                v = tuple(a + b for a, b in zip(u, delta))
                # Ignore directions where we fall off the array
                if not all(0 <= x < dim for x, dim in zip(v, m.shape)):
                    continue
                
                # Connect to neighbor if height increments by one
                otherval = m[v]
                if otherval == val + 1:
                    self.edges[u].append(v)
                    #
                #
            #
        #
    #
    
    def neighbors(self, u):
        return self.edges.get(u, [])
    
    def bfs(self, startat):
        """Runs breadth-first search, starting at the given node"""

        final_paths = []
        current_paths = [[startat]]
        
        # Keep going while current paths (no need to worry ab cycles because trails must increase in height)
        while current_paths:
            paths_next = []
            
            for path in current_paths:
                head = path[-1]
                neighbors = self.neighbors(head)
                
                # Extend by all neighbors. If none, the path is exhausted
                if neighbors:
                    paths_next += [path + [neighbor] for neighbor in neighbors]
                else:
                    final_paths.append(path)
                #
            
            current_paths = paths_next
        
        return final_paths
    #


def get_all_trails(G: Graph) -> list:
    """Returns a list of lists of all trails from all trailheads.
    Each element in the result is a list of trails from a single trailhead, ending at a peak."""

    # Get the nodes where trails start and stop
    startat = 0
    endat = 9
    trailheads = G.value_lookup[startat]
    sinks = set(G.value_lookup[endat])
    
    res = []
    for trailhead in trailheads:
        # Keep all trails from this trailhead where trail ends at a peak
        hikes = [hike for hike in G.bfs(startat=trailhead) if hike[-1] in sinks]
        res.append(hikes)
    
    return res


def sum_trailhead_scores(trail_groups: list):
    res = 0
    for hikes in trail_groups:
        peaks = {hike[-1] for hike in hikes}
        res += len(peaks)
        
    return res


def sum_trailhead_ratings(trail_groups: list):
    res = 0
    for hikes in trail_groups:
        distinct_trails = {tuple(hike) for hike in hikes}
        res += len(distinct_trails)
    
    return res


def solve(data: str):
    m = parse(data)
    G = Graph(m)
    trail_groups = get_all_trails(G=G)
    
    star1 = sum_trailhead_scores(trail_groups=trail_groups)
    print(f"Solution to part 1: {star1}")

    star2 = sum_trailhead_ratings(trail_groups=trail_groups)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2024, 10
    from aoc.utils.data import check_examples
    #check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
