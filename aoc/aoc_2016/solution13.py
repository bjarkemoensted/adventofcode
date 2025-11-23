# ·  .`  ·*`*+· .  `·   ·  ·*·  `+ . *   ·.· •     `·.··  •. *.` ·  · *· `.  ·+·
# `· ·*  •  .· .·`    ·  A Maze of Twisty Little Cubicles   +·`.·    * .··  `*·.
# .·`  .·*  ·  ·       https://adventofcode.com/2016/day/13  . ·` ·  ·.*`. ·*· ·
# *.*· •  ··`.  +· ·.· *·`·.  +·  `*.  .· · *`·.+ ·   ·*`·.    *• `·.  ·.+· .`· 


import heapq


def parse(s: str):
    res = int(s)
    return res


class Graph:
    def __init__(self, offset: int):
        self._offset = offset

    def coord_is_open(self, crd):
        x, y = crd
        if x < 0 or y < 0:
            return False

        # The space is open if some function of x and y, plus some offset, contains an equal number of ones in binary
        base_ = x*x + 3*x + 2*x*y + y + y*y
        sum_ = base_ + self._offset
        bin_ = str(bin(sum_))[2:]
        n_ones = bin_.count("1")
        res = n_ones % 2 == 0

        return res

    def get_neighbors(self, crd):
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        res = []
        for step in steps:
            newcrd = tuple(a+b for a, b in zip(step, crd))
            if self.coord_is_open(newcrd):
                res.append(newcrd)

        return res


def manhatten_dist(crd1, crd2):
    res = sum(abs(a - b) for a, b in zip(crd1, crd2))
    return res


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


def a_star(G: Graph, target, start=(1, 1)) -> list[tuple[int, int]]:
    open_ = Queue()
    d_g = dict()
    d_g[start] = 0

    f0 = manhatten_dist(start, target)

    open_.push(start, priority=f0)
    camefrom: dict[tuple[int, int], tuple[int, int]] = dict()

    while open_:
        current = open_.pop()
        if current == target:
            # reconstruct path
            path_rev = [current]
            while path_rev[-1] != start:
                path_rev.append(camefrom[path_rev[-1]])
            path = path_rev[::-1]
            return path

        for neighbor in G.get_neighbors(current):
            uv = 1  # dist between two sites
            g_tentative = d_g[current] + uv
            improved = g_tentative < d_g.get(neighbor, float("inf"))
            if improved:
                camefrom[neighbor] = current

                d_g[neighbor] = g_tentative
                f = g_tentative + manhatten_dist(neighbor, target)
                if neighbor not in open_:
                    open_.push(neighbor, priority=f)
                #
            #
        #
    raise RuntimeError("No path found")


def distinct_bfs_count(G: Graph, n_steps, start=(1, 1)):
    seen = {start}
    frontier = {start}

    for _ in range(n_steps):
        new_frontier = set([])
        for crd in frontier:
            for neighbor in G.get_neighbors(crd):
                if neighbor not in seen:
                    seen.add(neighbor)
                    new_frontier.add(neighbor)
                #
            #
        frontier = new_frontier

    res = len(seen)
    return res


def solve(data: str) -> tuple[int|str, int|str]:
    offset = parse(data)
    G = Graph(offset)

    target = (7, 4) if offset == 10 else (31, 39)

    path = a_star(G=G, target=target)
    star1 = len(path) - 1
    print(f"Getting to site {target} requires {star1} steps.")

    n_steps = 50
    star2 = distinct_bfs_count(G, n_steps)
    print(f"With at most {n_steps} steps, {star2} distinct sites can be reached.")

    return star1, star2


def main() -> None:
    year, day = 2016, 13
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
