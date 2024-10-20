#  .ꞏ* ⸳. ꞏ.•   .⸳ • ꞏ       ꞏ   ⸳ .`•. ⸳   `    ꞏ  ⸳ . •`⸳`.  ⸳ꞏ *   ` . +*.  *
#   *⸳.  `   *ꞏ ⸳. ꞏ+ Radioisotope Thermoelectric Generators ⸳ꞏ    ⸳  *. ꞏ ` *ꞏ⸳
# .*•.       ꞏ` +   .` https://adventofcode.com/2016/day/11      ꞏ ꞏ.` +⸳ .    .
#  ⸳.`ꞏ• . . `+⸳       ⸳*ꞏ .`+ ⸳. ` ꞏ •  ꞏ.`  +  ⸳ꞏ      .  ⸳ +ꞏ .  `•  ꞏ*⸳    `


from functools import cache
import heapq
from itertools import combinations
import re


def parse(s):
    res = dict()
    floor = 1
    for line in s.split("\n"):
        generators = re.findall(r"\ba (\S*) generator", line)
        microchips = re.findall(r"\ba (\S*)-compatible microchip", line)
        floor_data = dict(generators=generators, microchips=microchips)
        res[floor] = floor_data

        floor += 1

    return res


def tuple_from_list(arr):
    """Turns a list of lists into a tuple of tuples"""
    el, stuff = arr
    res = (el, tuple(tuple(floor) for floor in stuff))

    return res


@cache
def _floor_fries(floor_contents: tuple):
    """Returns a boolean indicating whether any microchips on the given floor fries. Floor contents has format:
    (generator_a, chip_a, generator_b, chip_b, ...)"""

    radiation = 0
    unshielded_chips = 0
    for i in range(0, len(floor_contents), 2):
        generator, microchip = floor_contents[i:i+2]
        radiation += generator
        unshielded_chips += microchip and not generator

    return radiation and unshielded_chips


@cache
def _take_allowed_cargo(floor_contents: tuple, n_min: int = 1, n_max: int = 2) -> list:
    """Returns list of tuples like (elevator contents, remaining floor contents)"""
    nonzero_inds = [i for i, val in enumerate(floor_contents) if val == 1]
    res = []
    n_items = len(floor_contents)
    for n in range(n_min, n_max+1):
        for comb in combinations(nonzero_inds, n):
            cargo = tuple(int(i in comb) for i in range(n_items))
            remaining = tuple(val - moved for val, moved in zip(floor_contents, cargo))
            if not _floor_fries(remaining):
                res.append((cargo, remaining))
            #
        #

    return res


def heuristic(state: tuple) -> int:
    """Lower bound on the number of steps required to move everything to the top floor."""

    # Get the distances for all items to the top floor
    elevator, stuff = state
    top = len(stuff) - 1
    dists = [top - n for n, floor in enumerate(stuff) for item in floor if item]

    a = dists[0]  # Assume we can move both the furthest items all the way to the top
    b = 2*sum(dists[2:])  # After that, only 1 item at a time can effectively be moved (elevator can't travel empty)
    res = a + b

    return res


class Graph:
    def __init__(self, data: dict):
        self.floors = sorted(data.keys())
        self.n_floors = len(self.floors)

        self.materials_set = set(sum((sum(v.values(), []) for v in data.values()), []))
        self.materials = tuple(sorted(self.materials_set))
        self.types = ("generators", "microchips")

        # Max number of initial letters such that the first n letters are distinct
        n_chars = min(i for i in range(max(map(len, self.materials)))
                      if len({m[:i] for m in self.materials}) == len(self.materials))

        order = [(m, t) for m in self.materials for t in self.types]

        self.basis = [f"{m[:n_chars]}{t[0]}".upper() for m, t in order]

        stuff = [[int(mat in data[floor][type_]) for mat, type_ in order] for floor in self.floors]
        elevator_start = 0
        state_list = [elevator_start, stuff]

        self.initial_state = tuple_from_list(state_list)

    def state_repr(self, state: tuple):
        """Represents a state as a string"""
        pad = 3*" "
        longest = max(len(b) for b in self.basis)
        el_rep = "E"
        nah = "."
        lines = []
        el, stuff = state
        for i, stuff in enumerate(stuff):
            elems = [f"F{self.floors[i]}", el_rep if el == i else nah]
            elems += [self.basis[j] if val else nah for j, val in enumerate(stuff)]
            line = pad.join([s + (longest - len(s))*" " for s in elems])
            lines.append(line)

        res = "\n".join(lines[::-1])
        return res

    def display_state(self, state: tuple):
        """Displays a state as a string, similar to in the problem description. For debugging and stuff"""
        s = self.state_repr(state=state)
        print(s)

    def get_neighbors(self, state: tuple):
        elevator, stuff = state
        floor_contents = stuff[elevator]
        res = []
        for new_floor in (elevator + 1, elevator - 1):
            if not (0 <= new_floor < self.n_floors):
                continue

            dest_content = stuff[new_floor]

            for cargo, remaining in _take_allowed_cargo(floor_contents):
                new_dest_content = tuple(a + b for a, b in zip(dest_content, cargo))
                if _floor_fries(new_dest_content):
                    continue
                new_stuff = tuple(
                    remaining if i == elevator else
                    new_dest_content if i == new_floor else
                    contents
                    for i, contents in enumerate(stuff)
                )

                res.append((new_floor, new_stuff))
            #
        return res


def _get_final_state(state: tuple) -> tuple:
    elevator, stuff = state
    n = len(stuff)
    res = (n-1, tuple(tuple(int(i == n - 1) for _ in floor) for i, floor in enumerate(stuff)))

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


def a_star(data: dict, maxiter=None):
    if maxiter is None:
        maxiter = float("inf")
    n_its = 0
    G = Graph(data)
    initial = G.initial_state
    final = _get_final_state(initial)

    open_ = Queue()
    best = None
    closest = float("inf")
    maxlen = 0

    d_g = {initial: 0}  # maps states to shortest known path to it
    f_initial = 0 + heuristic(initial)
    d_f = {initial: f_initial}  # Maps states to lower bound on path through the state

    open_.push(initial, priority=f_initial)
    camefrom = dict()

    while open_:
        n_its += 1

        current = open_.pop()

        if current == final:
            path_rev = [current]
            while path_rev[-1] != initial:
                path_rev.append(camefrom[path_rev[-1]])
            res = path_rev[::-1]
            print()
            return res

        for neighbor in G.get_neighbors(current):
            d_uv = 1
            g_tentative = d_g[current] + d_uv
            improvement = g_tentative < d_g.get(neighbor, float("inf"))
            if improvement:
                camefrom[neighbor] = current
                d_g[neighbor] = g_tentative
                h = heuristic(neighbor)
                closest = min(closest, h)
                f_new = g_tentative + h
                if neighbor == final:
                    best = g_tentative if best is None else min(best, g_tentative)

                d_f[neighbor] = f_new
                if neighbor not in open_:
                    open_.push(neighbor, priority=f_new)
                #
            #
        if n_its % 10000 == 0:
            msg = f"{n_its} iterations, lowest h score: {closest}. Frontier size: {len(open_)}."
            if best is not None:
                msg += f" Best found: {best}."
            maxlen = max(maxlen, len(msg))
            msg = msg + (maxlen - len(msg))*" "
            print(msg, end="\r")

        if n_its >= maxiter:
            break
        #
    return


def solve(data: str):
    data = parse(data)

    path = a_star(data=data, maxiter=100000)
    star1 = len(path) - 1
    print(f"It takes a minimum of {star1} steps to bring everything to the top floor")

    more_stuff = ["elerium", "dilithium"]
    for k in ("generators", "microchips"):
        data[1][k] += more_stuff

    path2 = a_star(data=data)
    star2 = len(path2) - 1 if path2 else None  # no example for part 2 so just return None
    print(f"With the extra stuff, it takes a minimum of {star2} steps to bring everything to the top floor")

    return star1, star2


def main():
    year, day = 2016, 11
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    #solve(raw)


if __name__ == '__main__':
    main()
