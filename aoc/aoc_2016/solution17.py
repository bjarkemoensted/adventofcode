from functools import cache
import hashlib
import heapq


def parse(s):
    res = s
    return res


dir2step = dict(
    D=(1, 0),
    U=(-1, 0),
    L=(0, -1),
    R=(0, 1)
)

_chars_allowed = set("0123456789abcdef")
_chars_open_door = set("bcdef")
_directions_order = ("U", "D", "L", "R")


@cache
def _coord_from_path(path: str, starting_point=(0, 0)) -> tuple:
    steps = [starting_point]
    steps += [dir2step[c] for c in path]

    res = tuple(map(sum, zip(*steps)))
    return res


def get_open_doors(hash_: str):
    if len(hash_) < len(_chars_allowed) or len(set(hash_) - _chars_allowed) > 0:
        raise ValueError(f"Invalid hash: {hash_}.")

    res = [dir_ for dir_, char in zip(_directions_order, hash_) if char in _chars_open_door]
    return res


class Graph:
    def __init__(self, passcode: str, shape=(4, 4)):
        self.passcode = passcode
        self.shape = shape

    def _md5(self, path):
        s = self.passcode + path
        hash_ = hashlib.md5(s.encode("utf-8")).hexdigest()
        return hash_

    def get_neighbors(self, path):
        crd = _coord_from_path(path)
        res = []
        hash_ = self._md5(path)
        for direction in get_open_doors(hash_):
            step = dir2step[direction]
            newcoord = tuple(a+b for a, b in zip(crd, step))
            inside_grid = all(0 <= c < lim for c, lim in zip(newcoord, self.shape))
            if inside_grid:
                res.append(path+direction)
            #
        return res

    @cache
    def lower_bound_distance(self, path: str, target_coord: tuple):
        crd = _coord_from_path(path)
        manhatten_dist = sum(abs(a - b) for a, b in zip(crd, target_coord))
        return manhatten_dist


class Queue:
    def __init__(self):
        self._items = []
        self._items_set = set([])

    def push(self, item, priority: int):
        if item in self:
            raise ValueError
        heapq.heappush(self._items, (priority, item))
        self._items_set.add(item)

    def pop(self):
        _, item = heapq.heappop(self._items)
        self._items_set.remove(item)
        return item

    def __contains__(self, item):
        return item in self._items_set

    def __len__(self):
        return len(self._items)


def a_star(G: Graph, start_path="", target_coord=None) -> list|None:
    """Standard A* except this uses paths as nodes."""

    if target_coord is None:
        target_coord = tuple(c - 1 for c in G.shape)

    d_g = dict()
    d_g[start_path] = 0

    d_f = dict()
    d_f[start_path] = G.lower_bound_distance(start_path, target_coord)

    open_ = Queue()
    open_.push(start_path, priority=d_f[start_path])

    while open_:
        current = open_.pop()
        current_coord = _coord_from_path(current)
        if current_coord == target_coord:
            return current  # No need to reconstruct path because we're operating on paths directly

        for neighbor in G.get_neighbors(path=current):
            d_uv = 1
            g_tentative = d_g[current] + d_uv
            improved = g_tentative < d_g.get(neighbor, float("inf"))
            if improved:
                d_g[neighbor] = g_tentative
                h = G.lower_bound_distance(path=neighbor, target_coord=target_coord)
                f = g_tentative + h
                if neighbor not in open_:
                    open_.push(neighbor, priority=f)
                #
            #
        #
    #


def longest_path(G: Graph, start_path="", target_coord=None) -> int|None:
    """Just use a brute force approach. Returns the length of the longest path."""

    if target_coord is None:
        target_coord = tuple(c - 1 for c in G.shape)

    longest = None
    frontier = [start_path]
    while frontier:
        next_frontier = []
        for path in frontier:
            crd = _coord_from_path(path)
            if crd == target_coord:
                longest = max(longest, len(path)) if longest is not None else len(path)
                continue
            for neighbor in G.get_neighbors(path=path):
                next_frontier.append(neighbor)
        frontier = next_frontier

    return longest


def solve(data: str):
    passcode = parse(data)

    G = Graph(passcode=passcode)
    star1 = a_star(G=G)
    print(f"The shortest path to the vault is: {star1}.")

    star2 = longest_path(G=G)
    print(f"The longest possible path to the vault contains {star2} steps.")

    return star1, star2


def main():
    year, day = 2016, 17
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
