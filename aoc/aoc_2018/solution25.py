#  .·*`+·`·.   `·    •  .   `·*· .·+  · .` *   · ·.*  · `  ` ·*. ·  `·+ *. `·* `
# ·`.` · * + ·   `.   ·  ·· Four-Dimensional Adventure  ··`.   +`*. ·  ·`  .·.  
# `·· *`. +·`.· `  ` · https://adventofcode.com/2018/day/25 * `·  `·. * ··` +`·.
#  ·`.·*`      ·.·`+   ·  .·*   ` ·.`  . ·  ` ·.* `  ··    *·`·   •* `· . ·*` .·

from collections import defaultdict

import numpy as np


def parse(s: str) -> np.ndarray:
    res = np.array([[int(v) for v in line.split(",")] for line in s.splitlines()])
    return res


def count_constellations(coords, max_dist: int=3) -> int:
    """Returns the number of constellations in the provided points.
    Computes all pairwise dists, and maps each coord to points within the threshold distance.
    We then start from every point, repeatedly adding nearby unassigned points until none remain."""
    
    # Find pairwise Manhatten dists, and pairs within max_dist of one another
    diffs = np.abs(coords[:, None] - coords[None, :])
    m = diffs.sum(axis=-1)
    connected = np.where(m <= max_dist)
    
    # Map index of each coordinate to set of nearby coords
    connections: dict[int, set[int]] = defaultdict(set)
    for i, j in zip(*(map(int, arr) for arr in connected)):
        connections[i].add(j)
    
    # Keep track of which inds have been assigned to a constellation
    assigned: set[int] = set()
    constellations = []
    
    for seed in range(len(coords)):
        # Skip ind if it's already been assigned
        if seed in assigned:
            continue
        
        # Start from the 'seed' and repeatedly add nearby unassigned points
        add_ = {seed}
        constellation: set[int] = set()
        
        while add_:
            # Add new points to current constellation, and register as assigned
            constellation |= add_
            assigned |= add_
            
            # Add unassigned nearby points to next iteration
            next_candidates = set.union(*(connections[node] for node in add_))
            add_ = next_candidates - assigned
        
        constellations.append(constellation)
    
    res = len(constellations)
    
    return res


def solve(data: str) -> tuple[int|str, None]:
    coords = parse(data)

    star1 = count_constellations(coords)
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 25
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
