# *  .ꞏ`   ꞏ    .* ⸳ `      ⸳ *.  `  ` ⸳•. ꞏ       ꞏ    ⸳*.  `*      *⸳ .+`ꞏ •`⸳
#   .• ꞏ     ` .  ꞏ   *⸳+. ` • Chronal Coordinates    ⸳ꞏ.     ꞏ.⸳  `  * ꞏ⸳• + .`
#  ⸳⸳ *  . ⸳ ꞏ  `*   ⸳ https://adventofcode.com/2018/day/6 .  ⸳ꞏ+  `ꞏ.     `⸳ꞏ*.
# ⸳.ꞏ  *   • .` +ꞏ * ꞏ⸳   `  ` *ꞏ.* ⸳ꞏ  *  ⸳ꞏ* `.  ⸳•.   ꞏ⸳ *.     +.` `⸳ꞏ   .⸳ 

from collections import Counter, defaultdict
import numpy as np
from typing import TypeAlias

coordtype: TypeAlias = tuple[int, int]


def parse(s):
    res = [tuple(map(int, line.split(", "))) for line in s.splitlines()]
    return res


def compute_dists(coords) -> np.typing.NDArray[np.int_]:
    """Compute a 3D matrix representing the shortest distance to each input coordinate."""
    
    # determine bounds
    y, x = zip(*coords)
    ya, yb = min(y), max(y)
    xa, xb = min(x), max(x)
    
    # Shift coordinates so they start at y, x = (0, 0)
    coords = [(i-ya, j-xa) for i, j in coords]
    yb -= ya
    xb -= xa
    
    # Make arrays of the x/y coordinates for computing Manhatten distances
    yx_shape = (yb+1, xb+1)
    y_coords, x_coords = np.indices(yx_shape)
    
    # Allocate array for holding the result
    z_inds = list(range(len(coords)))
    m = np.empty((*yx_shape, len(z_inds)), dtype=np.int_)
    
    # Make each z-'layer' represent the shortest distances (to A for z=0, B for z=1, etc)
    for k in z_inds:
        i0, j0 = coords[k]
        dists = np.abs(y_coords - i0) + np.abs(x_coords - j0)
        m[:,:,k] = dists
    
    return m


def get_largest_region(dists: np.typing.NDArray[np.int_]) -> int:
    """Returns the size of the largest cluster, disregarding infinite-size clusters."""

    # Determine shortest distance for each coordinate, and identify ties
    _min = dists.min(axis=2)
    match = dists == _min[:,:,np.newaxis]
    tied = (match.sum(axis=2) > 1)
    
    # Determine nearest input coordinates (0 for A, 1 for B, etc.)
    closest = dists.argmin(axis=2)
    # Identify clusters that touch the edges. Those will expand indefinitely and are ignored
    edges = [arr for ind in (0, -1) for arr in (closest[ind, :], closest[:, ind])]
    infinite_clusters = set(map(int, np.concatenate(edges)))
    
    # Determine size of clusters, considering only non-tied coordinates
    sizes = Counter(map(int, closest[~tied]))
    
    # Compute largest area among the finite clusters
    largest = max(sizes.keys(), key=lambda k: (k not in infinite_clusters, sizes[k]))
    res = sizes[largest]
    
    return res


def area_bounded_by_total_size(dists: np.typing.NDArray[np.int_], bound: int) -> int:
    """Returns the area bounded by the input distance."""
    
    # Simply compute total distance and count how many are below the max
    total_dists = dists.sum(axis=2)
    area = sum((total_dists < bound).flat)

    return area


def solve(data: str):
    coords = parse(data)
    coords_ij = [(y, x) for x, y in coords]
    
    
    dists = compute_dists(coords_ij)
    
    
    star1 = get_largest_region(dists)
    print(f"Solution to part 1: {star1}")

    bound = 32 if len(coords) < 10 else 10_000
    star2 = area_bounded_by_total_size(dists, bound=bound)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2018, 6
    from aoc.utils.data import check_examples
    
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
