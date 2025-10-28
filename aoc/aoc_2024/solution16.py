# *.·`. ·  *     ··`   `* . ·+.   ·   `·  *·   ·  .`*·*·   `.··*   ·+  `•. ·*`·.
# ·`·+   .·   `  .+·`.*·     ·*.` Reindeer Maze     ` . ·+`   . ·   `.··`*  .·  
# . *·` `·. ·     `+ · https://adventofcode.com/2024/day/16   ` .*  ·   .·  `  ·
# ·`.*· ·   •. · `  ·   .` *·  · `  .· ·*  .  ·+ .  · + `    `*. ·   ·  ·` *· .`


import networkx as nx
import numpy as np


dirs = (
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0)
)

# Map directions to the direction rotated left/right
rotations = {dir_: tuple(dirs[(i+shift) % len(dirs)] for shift in (-1, +1)) for i, dir_ in enumerate(dirs)}


def parse(s: str):
    m = np.array([list(line) for line in s.splitlines()])
    return m


def _weighted_edges(sites, step_cost: int, rotate_cost: int):
    """Iterates over connected 'states' (coordinate, direction) and the cost of transitioning from one
    to the other.
    Format for each result is (u, (v, cost)), with u and v having the format (coordinate, direction)"""
    
    nodes = set(sites)
    for node in sorted(sites):
        for dir_ in dirs:
            u = (node, dir_)
        
            # Add rotations
            for otherdir in rotations[dir_]:
                v = (node, otherdir)
                yield u, (v, rotate_cost)
            
            # Add forward step if possible
            forward = tuple(a+b for a, b in zip(node, dir_))
            if forward in nodes:
                v = (forward, dir_)
                yield u, (v, step_cost)
        #
    #


class Maze:
    start_char = "S"
    end_char = "E"
    space_char = "."
    
    weight = "cost"
    
    def __init__(self, m: np.ndarray, step_cost=1, rotate_cost=1000):
        # Determine indices of start/stop locations, and empty spaces
        chars = (self.start_char, self.end_char, self.space_char)
        d = {c: [tuple(map(int, locs)) for locs in np.argwhere(m == c)] for c in chars}
        assert all(len(v) == 1 for k, v in d.items() if k in (self.start_char, self.end_char))
        
        self.start = d[self.start_char][0]
        self.end = d[self.end_char][0]
        
        # Use all empty and start/stop nodes as part of the maze
        locations = sum((v for v in d.values()), [])
        self.G = nx.DiGraph()
        
        for u, (v, cost) in _weighted_edges(locations, step_cost=step_cost, rotate_cost=rotate_cost):
            self.G.add_edge(u, v, cost=cost)
        
    
    def score(self, path):
        """Returns the score of a given path"""
        res = nx.path_weight(G=self.G, path=path, weight=self.weight)
        return res

    def shortest_paths(self, start_site=None, target_site=None, start_dir=None):
        """Returns a list of the shortest path(s) from the start to the target site."""
        
        # Look for paths starting with the given start location and direction (east by default)
        start_dir = (0, 1) if start_dir is None else start_dir
        start_site = self.start if start_site is None else start_site
        source = (start_site, start_dir)
        
        # Target state is (target site, any direction), but skip directions that don't make sense
        target_site = self.end if target_site is None else target_site
        targets = []
        for dir_ in dirs:
            # If we don't end at the target by taking a step, the path must end with an unnecessary rotation
            prev = (tuple(a - b for a, b in zip(target_site, dir_, strict=True)), dir_)
            if prev in self.G:
                targets.append((target_site, dir_))
            #
        
        # Keep track of the paths with the lowest score seen so far
        record = float("inf")
        paths = []

        for target in targets:
            for path in nx.all_shortest_paths(G=self.G, source=source, target=target, weight=self.weight):
                # Discard previously found shortest paths if the record gets beat
                score = self.score(path)
                if score < record:
                    record = score
                    paths = []
                
                # Keep path if it matches the current record
                if score == record:
                    paths.append(path)
                #
            #
        
        return paths
    #


def solve(data: str) -> tuple[int|str, int|str]:
    m = parse(data)
    
    maze = Maze(m=m)

    paths = maze.shortest_paths()
    star1 = maze.score(paths[0])
    print(f"Solution to part 1: {star1}")

    nodes_along_best_paths = {site for site, _ in sum(paths, [])}
    star2 = len(nodes_along_best_paths)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 16
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
