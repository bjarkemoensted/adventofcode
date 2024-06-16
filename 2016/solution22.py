import heapq
import networkx as nx
import re


_test = """Filesystem            Size  Used  Avail  Use%
/dev/grid/node-x0-y0   10T    8T     2T   80%
/dev/grid/node-x0-y1   11T    6T     5T   54%
/dev/grid/node-x0-y2   32T   28T     4T   87%
/dev/grid/node-x1-y0    9T    7T     2T   77%
/dev/grid/node-x1-y1    8T    0T     8T    0%
/dev/grid/node-x1-y2   11T    7T     4T   63%
/dev/grid/node-x2-y0   10T    6T     4T   60%
/dev/grid/node-x2-y1    9T    8T     1T   88%
/dev/grid/node-x2-y2    9T    6T     3T   66%"""


def read_input():
    #return _test
    with open("input22.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
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
        vals = [node] + vals
        d = {k: v for k, v in zip(headers, vals)}
        data.append(d)

    ivals, jvals = map(set, zip(*[d["node"] for d in data]))
    assert all(0 in set_ for set_ in (ivals, jvals))
    nrows, ncols = max(ivals) + 1, max(jvals) + 1

    size_arr = [[0 for _ in range(ncols)] for _ in range(nrows)]
    used_arr = [[0 for _ in range(ncols)] for _ in range(nrows)]

    for d in data:
        i, j = d["node"]
        size_arr[i][j] = d["size"]
        used_arr[i][j] = d["used"]

    size_arr = _tuplify(size_arr)
    used_arr = _tuplify(used_arr)

    return used_arr, size_arr


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


def _infer_shape(arr: list|tuple):
    nrows = len(arr)
    ncols = len(arr[0])
    if any(len(row) != ncols for row in arr):
        raise ValueError(f"Got different number of elements in each row.")

    return nrows, ncols


def _tuplify(data: list|tuple):
    """Ensures data is a tuple. Assumes 2D array format."""
    res = tuple(tuple(row) for row in data)
    return res


def _determine_empty_site(used: tuple) -> tuple:
    """Returns the coord (i, j) of the empty site in the input array. Raises error if there's not exactly empty site."""
    zeroes = [(i, j) for i, row in enumerate(used) for j, val in enumerate(row) if val == 0]
    if len(zeroes) != 1:
        raise ValueError(f"Expected to find exactly 1 empty node, but found {len(zeroes)}: {zeroes}")

    res = zeroes[0]
    return res


def _format_elems(arr: list|tuple, pad=" ") -> str:
    """Helper method for padding and aligning string representations of the grid like on the webpage."""
    maxlen = max(map(len, [elem for row in arr for elem in row]))
    lines = []
    for row in arr:
        line_elems = []
        for s in row:
            elem = s + " "*(maxlen - len(s))
            line_elems.append(elem)

        lines.append(pad.join(line_elems))
    res = "\n".join(lines)

    return res


def _neighbor_sites(crd: tuple, shape: tuple):
    """Given a (i, j) coordinate and shape tuple, iterates over the 4 adjacent sites (or fewer if out of bound)"""
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for step in steps:
        newcrd = tuple(x + delta for x, delta in zip(crd, step))
        if all(0 <= coord < lim for coord, lim in zip(newcrd, shape)):
            yield newcrd


def _iterate_crd_and_vals(arr: list|tuple):
    """Given a 2d array, iterates over coordinate, value pairs like (i, j), val."""
    for i, row in enumerate(arr):
        for j, val in enumerate(row):
            yield (i, j), val


def simple_shortest_paths_lattice_with_walls(shape: tuple, walls: set) -> nx.Graph:
    """Constructs a regular 2D-lattice with the specified shape. walls is a set of coordinates (i, j) that are
    inaccessible and thus not connected to any other nodes."""

    G = nx.Graph()
    nrows, ncols = shape
    nodes = {(i, j) for j in range(ncols) for i in range(nrows)}
    G.add_nodes_from(nodes)
    for u in nodes:
        if u in walls:
            continue
        for v in _neighbor_sites(u, shape=shape):
            if v in walls:
                continue
            if v in nodes:
                G.add_edge(u, v)
            #
        #

    paths = nx.all_pairs_shortest_path(G)
    dists = {u: {v: len(p) - 1 for v, p in d.items()} for u, d in paths}

    return dists


class State:
    def __init__(self, used: tuple, goal_data_coord: tuple, empty_coord: tuple = None):
        """Represents a configuration of data on the grid.
        used is a tuple of tuples representing the amount of data stored at node (i, j)
        goal_data_coord is the (i, j) coordinate of the node currently holding the objective data.
        empty_coord is the coordinate of the currently empty node."""

        assert isinstance(used, tuple)
        self.used = used
        self.shape = _infer_shape(self.used)
        self.goal_data_coord = goal_data_coord
        if empty_coord is None:
            empty_coord = _determine_empty_site(used=self.used)
        self.empty_coord = empty_coord

    def __iter__(self):
        """Iterater over the important fields, for converting to tuple and hashing"""
        for elem in (self.used, self.goal_data_coord):
            yield elem

    def __hash__(self):
        """Just use the tuple representation for hashing so we can use states as keys"""
        res = hash(tuple(self))
        return res

    def __lt__(self, other):
        """Need to implement <= to be able to put these on heaps."""
        return tuple(self) < tuple(other)

    def __str__(self):
        s = f"Data: {self.goal_data_coord}. Empty: {self.empty_coord}"
        return s

    def __repr__(self):
        s = f"{self.goal_data_coord}, {self.empty_coord}"
        return s


def move_data(state: State, crd_from, crd_to):
    """Moves data from one node to another. Assumes sufficient free space."""
    i1, j1 = crd_from
    i2, j2 = crd_to
    assert state.used[i2][j2] == 0

    # Determine the data size at the involved nodes after moving
    replace_vals = {
        crd_to: state.used[i1][j1],
        crd_from: 0
    }

    # Determine the contents and locations of the empty and goal nodes after moving
    new_used = tuple(
        tuple(
            replace_vals.get((i, j), state.used[i][j]) for j, val in enumerate(row)
        )
        for i, row in enumerate(state.used)
    )
    new_empty = crd_from
    new_goal = crd_to if crd_from == state.goal_data_coord else state.goal_data_coord

    # Create the resulting state
    res = State(used=new_used, goal_data_coord=new_goal, empty_coord=new_empty)
    return res


class Grid:
    def __init__(self, size: tuple, walls=None, destination=(0, 0)):
        """Grid class representing the total capacities of each node.
        walls is a set of nodes from which data cannot be moved.
        destination is kept in this class to allow easier visualisation (to put parentheses around that node like on
        the AoC page, etc.)."""

        assert isinstance(size, tuple)
        self.size = size
        self.shape = _infer_shape(self.size)
        self.destination = destination
        if walls is None:
            walls = set([])
        self.walls = walls

        # Cache the shortest paths on the lattice
        self._simple_dists = simple_shortest_paths_lattice_with_walls(shape=self.shape, walls=self.walls)

    def _state_to_string(self, state: State, compact: bool):
        """Format the state + grid in a nice way that resemples the web page. Only for visualization."""

        elems = []
        for i, used_row in enumerate(state.used):
            row = []
            for j, used in enumerate(used_row):
                size = self.size[i][j]
                elem = None
                crd = (i, j)
                if compact:
                    elem = "."
                    if crd == state.goal_data_coord:
                        elem = "G"
                    elif crd == state.empty_coord:
                        elem = "_"
                    elif crd in self.walls:
                        elem = "#"
                else:
                    elem = f"{used}T/{size}T"
                    if crd == state.goal_data_coord:
                        elem = f"[{elem}]"
                    elif j == state.goal_data_coord[1]:
                        elem = " " + elem
                    #
                if crd == self.destination:
                    elem = f"({elem})"
                elif j == self.destination[1]:
                    elem = " " + elem

                row.append(elem)

            elems.append(row)

        res = _format_elems(elems)

        return res

    def view_state(self, state: State, compact=True):
        s = self._state_to_string(state=state, compact=compact)
        print(s)

    def heuristic(self, state: State) -> int:
        """Lower bound on the number of steps needed to move the data to the destination.
        First the free space must be moved adjacent to the data, then repeatedly, the data can be moved at most one step
        at a time closer to the destination, followed by 5 steps of moving the free spot around the data to free up the
        next site in the shortest path from data to destination."""

        # Find the number of steps in the shortest path from using all viable nodes
        goal_to_dest = self._simple_dists[state.goal_data_coord][self.destination]
        if goal_to_dest == 0:
            return 0

        # The free spot must be moved adjacent to data first
        empty_to_goal = self._simple_dists[state.empty_coord][state.goal_data_coord] - 1
        # Best case, the free spot is between data and destination. After that, it's 1 data move and 4 free space moves
        shift_steps = 1 + 5*(goal_to_dest - 1)

        res = empty_to_goal + shift_steps
        return res

    def get_neighbors(self, state: State):
        """Given a state instance, return a list of the states which can be reached from there in a single step, e.g.
        by running a single move operation. Note that 'neighbor' here does not mean a neighboring node on the
        grid/lattice, but the entire configuration of data which can be achieved with a single move operation."""

        res = []
        # Determine the available space on the empty node to which we're moving data
        copy_to = state.empty_coord
        i2, j2 = copy_to
        space = self.size[i2][j2]

        # Check each adjacent site in the grid to see if the data there fits on the empty site
        for copy_from in _neighbor_sites(copy_to, shape=state.shape):
            i1, j1 = copy_from
            data_fits_on_target = space >= state.used[i1][j1]
            if data_fits_on_target:
                neighbor = move_data(state=state, crd_from=copy_from, crd_to=copy_to)
                res.append(neighbor)
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


def a_star(grid: Grid, initial_state: State, maxiter=None, verbose=True):
    if maxiter is None:
        maxiter = float("inf")

    camefrom = dict()

    d_g = {initial_state: 0}

    d_f = {initial_state: grid.heuristic(initial_state)}
    closest = float("inf")

    open_ = Queue()
    open_.push(initial_state, priority=d_f[initial_state])
    n_its = 0
    msg_maxlen = 0

    while open_:
        current = open_.pop()
        done = current.goal_data_coord == grid.destination
        if done:
            path = [current]
            while path[-1] != initial_state:
                path.append(camefrom[path[-1]])
            path = path[::-1]
            if verbose:
                print()
            return path

        for neighbor in grid.get_neighbors(state=current):
            d_uv = 1
            g_tentative = d_g[current] + d_uv
            improved = g_tentative < d_g.get(neighbor, float("inf"))
            if improved:
                camefrom[neighbor] = current
                d_g[neighbor] = g_tentative
                h = grid.heuristic(neighbor)
                f = g_tentative + h
                if h < closest:
                    closest = h
                if neighbor not in open_:
                    open_.push(neighbor, priority=f)
                #
            #
        n_its += 1
        if verbose:
            msg = f"{n_its}: Considering {len(open_)} states. Best heuristic: {closest}."
            msg_maxlen = max(msg_maxlen, len(msg))
            msg = msg + (msg_maxlen - len(msg))*" "
            print(msg, end="\r")

        if n_its > maxiter:
            break

    if verbose:
        print()
    return None


def main():
    raw = read_input()
    used_arr, size_arr = parse(raw)

    viable_pairs = get_viable_pairs(used_arr=used_arr, size_arr=size_arr)
    star1 = len(viable_pairs)
    print(f"There are {star1} viable pairs")

    viable_nodes = set(sum(map(list, viable_pairs), []))
    nrows, ncols = _infer_shape(used_arr)
    walls = {crd for crd, _ in _iterate_crd_and_vals(used_arr) if crd not in viable_nodes}
    grid = Grid(size=size_arr, walls=walls)

    goal_data_coord = (0, ncols - 1)
    initial_state = State(used=used_arr, goal_data_coord=goal_data_coord)

    path = a_star(grid=grid, initial_state=initial_state)
    star2 = len(path) - 1
    print(f"Data can be moved in {star2} steps")


if __name__ == '__main__':
    main()
