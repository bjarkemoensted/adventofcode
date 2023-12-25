import numpy as np


def read_input():
    with open("input23.txt") as f:
        puzzle_input = f.read()

    return puzzle_input


def parse(s):
    M = np.array([list(line) for line in s.split("\n")])
    return M


up = (-1, 0)
right = (0, 1)
down = (1, 0)
left = (0, -1)
all_dirs = (up, right, down, left)


slopes = {
    "^": up,
    ">": right,
    "v": down,
    "<": left
}


def get_neighbors(crd, M):
    for step in all_dirs:
        newcrd = tuple(a + b for a, b in zip(crd, step))
        if all(0 <= x < lim for x, lim in zip(newcrd, M.shape)):
            yield newcrd


def get_next_possible_locations(crd, M, slippery, exclude=None):
    """Returns a list of sites we may travel to from input coordinate on M.
    slippery denotes whether the slopes are slippery and arrows must be followed.
    exclude is an optional iterable of sites to which travel is not allowed."""

    symbol = M[crd]
    if symbol == "#":
        raise ValueError

    res = None
    if slippery and symbol in slopes:
        step = slopes[symbol]
        newcrd = tuple(a + b for a, b in zip(crd, step))
        res = [newcrd]
    else:
        res = [newcrd for newcrd in get_neighbors(crd, M) if M[newcrd] != "#"]

    if exclude is not None:
        res = [elem for elem in res if elem not in exclude]

    return res


class Pathfinder:
    def __init__(self, M, slippery):
        self.M = M
        self.slippery = slippery

    def get_branching_points(self):
        """Provides the 'branching points' on M. A branching point is a point where travel in several directions
        is possible, e.g. a fork in the road."""
        rows, cols = self.M.shape
        for i in range(rows):
            for j in range(cols):
                try:
                    neighbors = list(get_next_possible_locations((i, j), self.M, slippery=self.slippery))
                    if len(neighbors) > 2:
                        yield i, j
                    #
                except ValueError:
                    continue

    def branch(self, crd):
        """Takes the coordinate of a branching point. Returns a lsit of paths going from that point, to all
        neighboring branching points."""

        # Start by taking one step in each possible location
        paths = [[crd, point] for point in get_next_possible_locations(crd, self.M, slippery=self.slippery)]
        for i in range(len(paths)):
            # As long as travel on this path is only possible in a single direction, take one more step
            while True:
                nc = list(get_next_possible_locations(paths[i][-1], self.M, slippery=self.slippery, exclude=paths[i]))
                if len(nc) == 1:
                    head = nc[0]
                    paths[i].append(head)
                else:
                    break
                #
            #

        # Some paths might be a dead end. Delete any path where the only possible next step is a tile already visited
        for i in range(len(paths) - 1, -1, -1):
            nc = list(get_next_possible_locations(paths[i][-1], self.M, slippery=self.slippery))
            bad = nc[0] in paths[i] and self.M[paths[i][-1]] != "."
            if bad:
                del paths[i]

        return paths

    def branch_points_dists(self, start, stop):
        """Builds a distance mapping between all the branching points on M.
        Returns a dict mapping each branching point to paths going from there to other branching points in a format like
        (destination, path_len), so like
        {source1: [dest1, len1, dest2, len2], ...}"""

        # Investigate all branching points as well as the start/stop points
        branch_points = self.get_branching_points()
        points = list(branch_points) + [start, stop]

        d = dict()
        for crd in points:
            paths = self.branch(crd)
            for path in paths:
                source = crd
                dest = path[-1]
                len_ = len(path) - 1
                d[source] = d.get(source, []) + [(dest, len_)]
            #
        return d

    def longest(self, start, stop):
        """Finds the longest path from start to stop.
        Works by repeatedly finding the longest paths to stop node using exactly n branches.
        When extending to n + 1 branches, the previous solutions can be used.
        At each step, the longest paths from each node to the stop node is retained for all unique subsets of paths,
        so if 3 paths, e.g. a -> b -> c -> stop,  a -> c -> b -> stop and a -> d -> c -> stop are found, only the
        longest of the paths involving a, b, and c is retained."""

        res = float("-inf")

        # When 0 branches are used, the only solution is starting at the stopping node
        nullpaths = {(stop,): 0}
        paths_to_target = {stop: nullpaths}

        # Map all branching points to their neighboring branching points
        bd = self.branch_points_dists(start, stop)

        for i in range(1, len(bd)):
            print(f"*** Considering paths with exactly {i} line segments ***", end="\r")

            # Look at all paths from a branching point to its neighbor
            d_next = dict()
            for source, data in bd.items():
                new_paths = dict()
                for dest, dist in data:
                    # If there's no dest -> stop path with n-1 steps, there's no source -> stop with n steps.
                    if dest not in paths_to_target:
                        continue

                    # Get the longest paths from dest -> stop using n-1 steps
                    oldpaths = paths_to_target[dest]
                    for old_nodes, old_dist in oldpaths.items():
                        # We can't revisit any locations, so if source is already in the n-1 path, move to the next path
                        if source in old_nodes:
                            continue

                        # Found a new n step path from source -> stop
                        new_dist = old_dist + dist
                        if source == start:
                            # If the source node is the starting node, update result
                            res = max(res, new_dist)

                        # Update n step paths from this source. Use sorted tuple of visited nodes as key
                        new_nodes = tuple(sorted(list(old_nodes) + [source]))
                        new_paths[new_nodes] = max(new_paths.get(new_nodes, float("-inf")), new_dist)

                # Retain these paths for the next (n + 1 step) paths
                if new_paths:
                    d_next[source] = new_paths

            paths_to_target = d_next
        print()

        return res


def main():
    raw = read_input()
    M = parse(raw)

    start = (0, list(M[0]).index("."))
    stop = (M.shape[0] - 1, list(M[-1]).index("."))

    pathfinder = Pathfinder(M, slippery=True)
    star1 = pathfinder.longest(start, stop)
    print(f"Longest path has {star1} steps.")

    pathfinder = Pathfinder(M, slippery=False)
    star2 = pathfinder.longest(start, stop)
    print(f"Longest path without slippery slopes has {star2} steps.")


if __name__ == '__main__':
    main()
