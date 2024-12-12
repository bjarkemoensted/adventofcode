# .•. ꞏ*`ꞏ⸳  *.      ꞏ• `⸳ꞏ     •**ꞏ   ꞏ   ⸳  *`. ꞏ⸳*.`  ⸳+   ⸳• . ⸳ꞏ*ꞏ⸳`.  * ⸳ꞏ
# *  ꞏ .ꞏ .* `*  ⸳.  +ꞏ    ⸳ ꞏ .ꞏ Garden Groups ` ⸳.ꞏ*.  `*ꞏ⸳ .ꞏ*      .⸳*` ⸳.`⸳
# ⸳ *.  +.ꞏ    ꞏ  ⸳ •. https://adventofcode.com/2024/day/12    .⸳+•+` .*ꞏ ꞏ`   *
# `.`•ꞏ. *     ⸳*ꞏ+  ⸳ ꞏ . •⸳. • ꞏ. +` ꞏ.  `⸳* . ⸳ •ꞏ.•   ꞏ.` *    ⸳. *ꞏ .*⸳` •.


from collections import defaultdict
import numpy as np


def parse(s):
    m = np.array([list(line) for line in s.splitlines()])
    return m


def _get_neighbors(ind: tuple):
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    for dir_ in dirs:
        yield tuple(a + b for a, b in zip(dir_, ind))
    #


def get_regions(m):
    """Returns a list of regions, each represented as a set of indices (i, j) of plots in the region"""
    
    res = []
    allocated = set([])
    
    for ind in np.ndindex(m.shape):
        # Move on if index is already assigned to a region
        if ind in allocated:
            continue
        
        # Otherwise, create new region and repeatedly add neighors with the same plant type
        char = m[ind]
        region = set([])
        frontier = {ind}
        
        while frontier:
            region |= frontier
            next_ = set([])
            
            for oldind in frontier:
                for neighbor in _get_neighbors(oldind):
                    # Ignore neighboring plots outside the array
                    if not all(0 <= x < dim for x, dim in zip(neighbor, m.shape)):
                        continue
                    # If neighbor has the same plant as region and hasn't been added yet, add it
                    if m[neighbor] == char and neighbor not in region:
                        next_.add(neighbor)
                    #
                #
            frontier = next_
        
        res.append(region)
        allocated |= region
    
    return res


def iterate_intervals(numbers, maxdist=1):
    """Takes an iterable of numbers.
    Generates tuples (a, b) denoting lower and upper bounds on intervals in the numbers, e.g.
    [1,2,3,5,6,9] -> (1, 4), (5, 7), (9, 10)"""

    numbers = sorted(numbers)
    if not numbers:
        return
    
    def small_inc(ind):
        """Helper method to determine if the number at an index is followed by a number that's a little higher"""
        try:
            # Check if the next number is sufficiently close
            inc = numbers[ind+1] - numbers[ind]
            return inc <= maxdist
        except IndexError:
            # If there's no next number, return False
            return False
    
    i = 0
    
    while i < len(numbers):
        lower = numbers[i]
        
        # Scan forward as long as i -> i+1 is a small change
        while small_inc(i):
            i += 1
        
        # Done with current interval
        upper = numbers[i] + 1
        yield lower, upper
        
        # Proceed at next index
        i += 1
    #


def scan_lines(region, flip=False):
    """Iterates over sets of column indices j for a collection of i, j tuples.
    One set for each row (i) is returned.
    Rows are assumed to be coherent, e.g. only increments in steps of 1.
    If flip is True, iterates over rows instead of columns."""

    inds = region
    if flip:
        inds = [(j, i) for i, j in inds]
    
    # Group column inds j by rows
    d = defaultdict(lambda: set([]))
    for i, j in inds:
        d[i].add(j)
    
    # Ensure there are no gaps in the rows
    keys = sorted(d.keys())
    assert all(keys[i+1] - keys[i] == 1 for i in range(len(keys) - 1))
    
    for k in keys:
        yield d[k]
    #


def iterate_line_segments(region):
    """Given a collection of indices, iterates over all line segments surrounding the region.
    Generates tuples (a, b) representing the lower (inclusive) and upper (exclusive) bounds
    on border segments."""

    # Need both vertical and horizontal segments
    for flip in (False, True):
        # Get the lines we scan across and pad with empty sets
        lines = list(scan_lines(region=region, flip=flip))
        lines = [set([])] + lines + [set([])]
        
        for i in range(len(lines) - 1):
            # Get the indices at 2 consecutive lines
            a = lines[i]
            b = lines[i+1]
            
            # Generate line segments for the indices added/removed at this line
            for delta in (a-b, b-a):
                yield from iterate_intervals(delta)
            #
        #
    #


def fence_price(regions, segment_price=False):
    """Computes the price of fences around the input regions.
    If segment_price is True, each segments counts as having length 1."""

    res = 0
    for region in regions:
        area = len(region)
        # Compute the perimeter
        segments = iterate_line_segments(region=region)
        lens = (1 if segment_price else b - a for a, b in segments)
        perimeter = sum(lens)
        
        region_price = area*perimeter
        res += region_price
    
    return res


def solve(data: str):
    m = parse(data)
    regions = get_regions(m)
    
    star1 = fence_price(regions)
    print(f"Solution to part 1: {star1}")

    star2 = fence_price(regions, segment_price=True)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2024, 12
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
