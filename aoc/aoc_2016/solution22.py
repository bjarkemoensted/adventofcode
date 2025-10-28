# · ` *·· .·`· ` . ·    `·+ ·`  · +·.· .  ·`    *· ·`• ·. · ·  . · `· • *· `.··*
# `.··`+ *` · +·.·  ` ·*·` ·  ·.* Grid Computing ·.+ .· ` · `·`  .·  .·` * ·+·`.
# ·`  ·.`··•    ·`  .· https://adventofcode.com/2016/day/22   ·+··   `.·    ·. ·
# ·.`  ·  +·  ·.  ·`·  · *` .· ·`    +·   `· .  ·* ` ·   ·   `*·  .··*   ` .·*.`


from abc import ABCMeta, abstractmethod
import heapq
import networkx as nx
import re
from typing import cast, TypeAlias


coordtype: TypeAlias = tuple[int, int]


def _tuplify(data: list|tuple):
    """Ensures data is a tuple. Assumes 2D array format."""
    res = tuple(tuple(row) for row in data)
    return res


def parse(s: str):
    """Parses the input data into two 2D-arrays, where the value at coordinate (i, j) represents the used space and
    total space, respectively, at that node. Note that the x-y format of the input data is changed to (i, j) so the
    first and second coordinate denotes the row and column index, respectively."""

    data = []
    pattern = r"\/dev/grid/node-x(\d+)-y(\d+)\W+(\d+)T\W+(\d+)T\W+(\d+)T\W+(\d+)%"
    for line in s.split("\n"):
        m = re.match(pattern, line)
        headers = ("node", "size", "used", "avail", "use")
        if m is None:
            continue
        matches = m.groups()
        vals = [int(ms) for ms in matches]
        node_xy = (vals.pop(0), vals.pop(0))
        node = node_xy[::-1]
        usevals = [node] + vals
        d = {k: v for k, v in zip(headers, usevals, strict=True)}
        data.append(d)

    ivals, jvals = map(set, zip(*[d["node"] for d in data]))
    assert all(0 in set_ for set_ in (ivals, jvals))
    nrows, ncols = max(ivals) + 1, max(jvals) + 1

    size_arr = [[0 for _ in range(ncols)] for _ in range(nrows)]
    used_arr = [[0 for _ in range(ncols)] for _ in range(nrows)]

    for d in data:
        i, j = cast(coordtype, d["node"])

        size_arr[i][j] = cast(int, d["size"])
        used_arr[i][j] = cast(int, d["used"])

    size_arr = _tuplify(size_arr)
    used_arr = _tuplify(used_arr)

    return used_arr, size_arr


def _neighbor_sites(crd: tuple, shape: tuple):
    """Given a (i, j) coordinate and shape tuple, iterates over the 4 adjacent sites (or fewer if out of bounds)"""
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for step in steps:
        newcrd = tuple(x + delta for x, delta in zip(crd, step))
        if all(0 <= coord < lim for coord, lim in zip(newcrd, shape)):
            yield newcrd
        #
    #


def _iterate_crd_and_vals(arr: list|tuple):
    """Given a 2d array, iterates over coordinate, value pairs like (i, j), val."""
    for i, row in enumerate(arr):
        for j, val in enumerate(row):
            yield (i, j), val
        #
    #


def _infer_shape(arr: list|tuple):
    nrows = len(arr)
    ncols = len(arr[0])
    if any(len(row) != ncols for row in arr):
        raise ValueError(f"Got different number of elements in each row.")

    return nrows, ncols


def _determine_empty_site(used: tuple) -> tuple:
    """Returns the coord (i, j) of the empty site in the input array. Raises error if there's not exactly empty site."""
    zeroes = [(i, j) for i, row in enumerate(used) for j, val in enumerate(row) if val == 0]
    if len(zeroes) != 1:
        raise ValueError(f"Expected to find exactly 1 empty node, but found {len(zeroes)}: {zeroes}")

    res = zeroes[0]
    return res


class Grid:
    """Grid class for representing the computing grid.
    Constructs all shortest pairs between all viable nodes and stores them.
    Exposes helper methods for getting the shortest path between any two viable nodes,
    the distance of said paths, and a set of nodes on paths.
    
    The idea is that this class represents the 'gameboard' on which various states, defined by
    the location of the empty node and the goal data, are defined."""

    def __init__(self, viable_nodes, shape: tuple, target_node=(0, 0)):
        self.shape = shape
        self.nodes = {(i, j) for (i, j) in viable_nodes}
        self.neighbors: dict[coordtype, set[coordtype]] = dict()
        self.target_node = target_node
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        for u in self.nodes:
            for v in _neighbor_sites(u, shape=shape):
                if v in self.nodes:
                    G.add_edge(u, v)
                    self.neighbors[u] = self.neighbors.get(u, set([])) | {v}
                #
            #

        self.paths = {u: {v: p for v, p in path_.items()} for u, path_ in nx.all_pairs_shortest_path(G)}
        
    def path(self, u, v):
        """Returns shortest path from u -> v"""
        res = self.paths[u][v]
        return res
    
    def nodes_on_shortest_paths(self, goal, empty=None):
        res = set(self.path(goal, self.target_node))
        if empty is not None:
            res |= set(self.path(empty, goal))
        
        return res
    
    def dist(self, u, v):
        """Returns the shortest distance from u -> v"""
        p = self.path(u=u, v=v)
        res = len(p) - 1
        return res


class State(metaclass=ABCMeta):
    char_goal = "G"
    char_path = "*"
    char_space = "·"
    char_wall = "#"
    char_empty = "_"

    def __init__(self, empty: tuple, goal: tuple):
        self.empty = empty
        self.goal = goal

    @property
    @abstractmethod
    def grid(self) -> Grid:
        pass

    def _determine_rep(self, i, j):
        coord = (i, j)
        
        path = self.grid.nodes_on_shortest_paths(empty=self.empty, goal=self.goal)
        
        char = self.char_space if coord in self.grid.nodes else self.char_wall
        if coord == self.empty:
            char = self.char_empty
        elif coord == self.goal:
            char = self.char_goal
        elif coord in path:
            char = self.char_path
        
        res = f"({char})" if coord == self.grid.target_node else f" {char} "
        return res

    def neighbors(self):
        con = type(self)
        for empty in self.grid.neighbors.get(self.empty, set([])):
            goal = self.empty if empty == self.goal else self.goal
            yield con(empty=empty, goal=goal)
        #
    
    def heuristic(self):
        goal_to_dest = self.grid.dist(self.goal, self.grid.target_node)
        if goal_to_dest == 0:
            return goal_to_dest
    
        n_steps_to_goal_neighbor = self.grid.dist(self.empty, self.goal) - 1

        n_shifts_lower_bound = 1 + 5*(goal_to_dest - 1)
        res = n_steps_to_goal_neighbor + n_shifts_lower_bound
        
        return res

    def _as_tuple(self):
        res = (self.empty, self.goal)
        return res

    def __lt__(self, other):
        return self._as_tuple() < other._as_tuple()

    def __str__(self):
        lines = []
        rows, cols = self.grid.shape
        for i in range(rows):
            line = " ".join([self._determine_rep(i, j) for j in range(cols)])
            lines.append(line)
        
        res = "\n".join(lines)
        return res


def get_viable_pairs(used_arr, size_arr):
    """Takes arrays representing used and total space at each node.
    Returns a list of pairs of coordinates [((i1, j1), (i2, j2)), ...] representing 'viable pairs' of nodes.
    Nodes that are not viable are the 'wall nodes'."""

    res = []
    for crd_a, used_a in _iterate_crd_and_vals(used_arr):
        for crd_b, used_b in _iterate_crd_and_vals(used_arr):
            ib, jb = crd_b
            avail = size_arr[ib][jb] - used_b
            if crd_a == crd_b:
                continue
            if used_a == 0:
                continue
            fits = used_a <= avail
            if fits:
                res.append((crd_a, crd_b))
            #
        #

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

    def __str__(self):
        s = f"Priority queue thingy with {len(self)} elements."
        return s



def a_star(initial_state: State):
    openset = Queue()

    camefrom: dict[State, State] = dict()

    def reconstruct(head: State) -> list[State]:
        nonlocal camefrom
        rev = [head]
        while head in camefrom:
            head = camefrom[head]
            rev.append(head)
        
        res = rev[::-1]
        return res
    
    h = initial_state.heuristic()
    dist = 0
    f_score = h + dist
    g = {initial_state: dist}  # g[u] is shortest path from start u
    f = {initial_state : f_score}  # f[u] is a lower bound on path through u
    openset.push(initial_state, priority=f_score)

    record = float("inf")
    
    while openset:
        u = openset.pop()
        if u.heuristic() == 0:
            res = reconstruct(u)
            return res
        
        for v in u.neighbors():
            d = 1
            g_tentative = g[u] + d
            improvement = g_tentative < g.get(v, float("inf"))
            if improvement:
                camefrom[v] = u
                g[v] = g_tentative
                h = v.heuristic()
                f_score = g_tentative + h
                f[v] = f_score
                openset.push(v, priority=f_score)
                record = min(record, h)
            #
        #
    #


def solve(data: str) -> tuple[int|str, int|str]:
    used_arr, size_arr = parse(data)
    
    viable_pairs = get_viable_pairs(used_arr=used_arr, size_arr=size_arr)
    star1 = len(viable_pairs)
    print(f"There are {star1} viable pairs")

    shape = _infer_shape(used_arr)
    _, ncols = shape
    viable_nodes = set(sum(map(list, viable_pairs), []))
    grid = Grid(viable_nodes=viable_nodes, shape=shape)

    goal_data_coord = (0, ncols - 1)
    empty_site_coord = _determine_empty_site(used_arr)

    GridState = type("GridState", (State,), {"grid": grid})
    initial_state = GridState(empty=empty_site_coord, goal=goal_data_coord)
    path = a_star(initial_state)


    star2 = len(path) - 1
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2016, 22
    from aocd import get_data
    
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
