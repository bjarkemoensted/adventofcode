from collections import defaultdict
from functools import cache
from itertools import product
import numpy as np


_test = """        ...#
        .#..
        #...
        ....
...#.......#
........#...
..#....#....
..........#.
        ...#....
        .....#..
        .#......
        ......#.

10R5L5R10L4R5L5"""


def read_input():
    #return _test
    with open("input22.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    map_, instructions_raw = s.split("\n\n")

    map_ = map_.split("\n")
    rows = len(map_)
    cols = max([len(line) for line in map_])

    M = np.array([[" " for _ in range(cols)] for _ in range(rows)])
    for i, line in enumerate(map_):
        for j, char in enumerate(line):
            M[i, j] = char

    instructions = []
    buffer = ""
    for char in instructions_raw:
        if char in "LR":
            if buffer:
                instructions.append(int(buffer))
                buffer = ""
            instructions.append(char)
        else:
            buffer += char
        #
    if buffer:
        instructions.append(int(buffer))

    return M, instructions


def _add_tuples(a, b):
    assert len(a) == len(b)
    res = tuple(c1 + c2 for c1, c2 in zip(a, b))
    return res


def _multiply_tuple(n: int, tup: tuple):
    res = tuple(n*x for x in tup)
    return res


def _neighbor_offsets(dim: int=None, include_diagnonal=True, initial_pos=None):
    """Generates n-dimensional offsets like (0, 1), ... for finding neighbor sites."""
    res = []
    if initial_pos is None:
        initial_pos = tuple(0 for _ in range(dim))
    elif dim is None:
        dim = len(initial_pos)

    shifts = (-1, 0, +1)
    for tup in product(shifts, repeat=dim):
        n_nonzero = sum(x != 0 for x in tup)
        if n_nonzero == 0 or n_nonzero > 1 and not include_diagnonal:
            continue
        coord = tuple(x + delta for x, delta in zip(initial_pos, tup))
        res.append(coord)

    res = tuple(res)
    return res


def _iterate_neighbors(shape: tuple, include_diagnonal=True):
    """Generates pairs of coords and neighbor coords."""
    shifts = _neighbor_offsets(dim=len(shape), include_diagnonal=include_diagnonal)
    for coord in np.ndindex(*shape):
        neighbors = tuple(tuple(x + delta for x, delta in zip(coord, shift)) for shift in shifts)
        yield coord, neighbors


@cache
def _rotate_direction_tuple(dir_: tuple, turn: str):
    """Takes a direction tuple (i, j) format. Returns the corresponding direction tuple after turing 90 degrees
    left/right (turn="L"/"R")."""

    vec = np.array(dir_)
    # Rotation matrix for 90 degree turn in positive direction
    R = np.array([
        [0, -1],
        [1, 0]
    ])
    turn2fac = {"L": 1, "R": -1}
    factor = turn2fac[turn]
    newvec = (factor*R).dot(vec)
    res = tuple(newvec)
    return res


class State:
    right = (0, 1)
    down = (1, 0)
    left = (0, -1)
    up = (-1, 0)

    _directions = (right, down, left, up)

    _direction2symbol = {
        right: '>',
        down: 'v',
        left: '<',
        up: '^'
    }

    @property
    def facing_symbol(self):
        char = self.custom_symbol
        if char is None:
            char = self._direction2symbol.get(self.dir_, "o")
        return char

    def __init__(self, pos: tuple, dir_: tuple | None, custom_symbol: str = None):
        if not (dir_ is None or dir_ in self._directions):
            raise ValueError(f"{dir_} is not a valid direction")
        if isinstance(custom_symbol, str) and len(custom_symbol) != 1:
            raise ValueError("Nah")

        self.pos = pos
        self.dir_ = dir_
        self.custom_symbol = custom_symbol

    def rotate(self, turn: str):
        if self.dir_ is None:
            raise ValueError
        newdir = _rotate_direction_tuple(dir_=self.dir_, turn=turn)
        res = type(self)(pos=self.pos, dir_=newdir)
        return res

    def step(self):
        if self.dir_ is None:
            raise ValueError
        newpos = tuple(x + delta for x, delta in zip(self.pos, self.dir_))
        res = type(self)(pos=newpos, dir_=self.dir_)
        return res

    def reverse(self):
        if self.dir_ is None:
            raise ValueError
        newdir = _multiply_tuple(-1, self.dir_)
        res = type(self)(pos=self.pos, dir_=newdir)
        return res

    def __repr__(self):
        return f"{self.pos} {self.facing_symbol}"

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for tup in (self.pos, self.dir_):
            yield tup

    def __eq__(self, other):
        return type(self) is type(other) and tuple(self) == tuple(other)

    def __hash__(self):
        return hash(tuple(self))


class Map:
    free_tile = "."
    wall_tile = "#"
    empty_tile = " "

    def __init__(self, layout: np.array):
        self.layout = layout.copy()
        allowed_chars = {self.free_tile, self.wall_tile, self.empty_tile}
        bad_chars = {char for char in layout.flat if char not in allowed_chars}
        if bad_chars:
            raise ValueError(f"Could'nt recognize all characters in layout: {', '.join(sorted(bad_chars))}")

    def get_initial_state(self, facing=(0, 1)):
        """Builds the initial state: position at leftmost free til on the top row. Facing east by default."""
        for coord, char in np.ndenumerate(self.layout):
            if char == ".":
                res = State(pos=coord, dir_=facing)
                return res

    def position_is_off_map(self, pos: tuple) -> bool:
        """Takes a state and returns a bool indicating if the state represents a site not on the map."""
        if len(pos) != len(self.layout.shape):
            raise ValueError
        out_of_bounds = not all(0 <= x < lim for x, lim in zip(pos, self.layout.shape))
        if out_of_bounds:
            return True

        falloff = self.layout[pos] == self.empty_tile
        return falloff

    def _reset_coords(self, state: State):
        """Resets coordinates using periodic boundaries, so like if shape = (5, 5), then -1 -> 4 and 5 > 0."""
        pos = tuple(x % lim for x, lim in zip(state.pos, self.layout.shape))
        res = State(pos=pos, dir_=state.dir_)
        return res

    def wraparound(self, state):
        """When taking a step forward falls off the map, the method returns the resulting state after wrapping"""
        offmap = state.step()
        if not self.position_is_off_map(offmap.pos):
            raise ValueError  # shouldn't happen but check to be sure

        state = self._reset_coords(offmap)
        while self.position_is_off_map(state.pos):
            state = self._reset_coords(state.step())
        return state

    def attempt_single_step(self, state: State):
        """Attempts a single step forward. Wraps around if falling off the map, and stops if hitting a wall"""
        newstate = state.step()
        if self.position_is_off_map(newstate.pos):
            newstate = self.wraparound(state)

        hits_wall = self.layout[newstate.pos] == self.wall_tile
        if hits_wall:
            return state
        return newstate

    def step(self, state: State, n_steps: int):
        """Takes the input number of steps"""
        for _ in range(n_steps):
            newstate = self.attempt_single_step(state)
            if newstate == state:
                break
            state = newstate
        return newstate

    def move(self, state: State, instruction: str | int) -> State:
        """Follows a given instruction (turning left right of taking a number of steps forward)"""
        if instruction in ("L", "R"):
            return state.rotate(turn=instruction)
        elif isinstance(instruction, int):
            return self.step(state=state, n_steps=instruction)
        else:
            raise ValueError

    def display(self, states: list | State = None, print_=True):
        """Helper method for displaying the map with a number or states"""
        if isinstance(states, State):
            states = [states]
        if states is None:
            states = []
        state_chars = {state.pos: state.facing_symbol for state in states}

        lines = []
        for i, row in enumerate(self.layout):
            line = ""
            for j, map_char in enumerate(row):
                char = state_chars.get((i, j), map_char)
                line += char
            lines.append(line)
        res = "\n".join(lines)
        if print_:
            return print(res)
        return res

    def __str__(self):
        res = self.display(states=None, print_=False)
        return res

    def __getitem__(self, item):
        if isinstance(item, State):
            item = item.pos
        if isinstance(item, (tuple, int)):
            return self.layout[item]


class Cube(Map):
    def __init__(self, layout: np.array):
        super().__init__(layout=layout)

        self.wrapmap = self._zip_walk()

    def _zip_walk(self):
        """Attempt at automatically identifying which edges pair with which when folding the cube.
        Works by identifying all states from which taking a step forward will fall off the map.
        These are then mapped to the state one should arrive at after wrapping around the cube.

        This works by identifying the inner (convex) corners of the map. From each corner a pair of paths are initiated.
        The paths extend incrementally, by repeatedly adding the neighboring state.
        If an outer (concave) corner is added to both paths in a pair, the paths stop growing.
        When all such pairs of paths have finished growing, the states from each pair of paths are zipped together.
        This results in a mapping from states falling off to states arriving after wrapping around the cube."""

        pos2falloffs = defaultdict(lambda: [])  # map positions to the states falling of at that location
        fallsite2states = defaultdict(lambda: [])  # Keep track of where those states point to on the empty tiles

        # Go over all the positions and itentify falloff states
        for pos in np.ndindex(self.layout.shape):
            if self.position_is_off_map(pos):
                continue
            for shift in _neighbor_offsets(dim=len(self.layout.shape), include_diagnonal=False):
                neighbor = _add_tuples(pos, shift)
                if self.position_is_off_map(neighbor):
                    falloff_state = State(pos=pos, dir_=shift)
                    pos2falloffs[pos].append(falloff_state)
                    fallsite2states[neighbor].append(falloff_state)
                #
            #

        # Define pairs of paths, starting with the pairs of nodes adjacent to the concave corners
        path_pairs = []
        for v in fallsite2states.values():
            if len(v) > 1:
                path = [[start_state] for start_state in v]
                path_pairs.append(path)

        def get_next_state(path):
            """Takes a path (iterable of falloff states) and returns the next state when traversing the boundary."""

            # Check for previously unseen states located at or around the most recently added state
            head = path[-1]
            candidate_positions = list(_neighbor_offsets(include_diagnonal=False, initial_pos=head.pos)) + [head.pos]
            neighbor_states = sum([pos2falloffs[pos] for pos in candidate_positions], [])
            candidates = [st for st in neighbor_states if st not in path]

            # If at a corner, pick the state matching the recent one on position or direction
            if len(candidates) > 1:
                candidates = [c for c in candidates if head.dir_ == c.dir_ or head.pos == c.pos]

            if len(candidates) != 1:
                raise ValueError

            return candidates[0]

        seen = set([])
        res = dict()

        while path_pairs:
            next_ = []
            for pair in path_pairs:
                # Extend pair of paths with one state
                next_pair = []
                for path in pair:
                    next_state = get_next_state(path)

                    # Ensure state hasn't been added to another path (shouldn't happen)
                    if next_state in seen:
                        raise ValueError
                    seen.add(next_state)

                    # Add the updated pair
                    next_path = path + [next_state]
                    next_pair.append(next_path)

                # Terminate pair-path if both the newly added nodes are at convex corners
                at_outer = [len(pos2falloffs[path[-1].pos]) > 1 for path in next_pair]
                if all(at_outer):
                    # Go over the nodes in the path and map them to each other with reversed direction
                    for a, b in zip(*pair):
                        res[a] = b.reverse()
                        res[b] = a.reverse()
                    #
                # Otherwise, keep iterating
                else:
                    next_.append(next_pair)
                #

            # Update the path pair list for the next iteration
            path_pairs = next_

        return res

    def wraparound(self, state):
        return self.wrapmap[state]


def make_full_path(map_: Map, instructions: list) -> list:
    """Runs the instructions on the given map instance, and returns a list of states which occur when
    executing the instructions."""
    instructions = sum([ins*[1] if isinstance(ins, int) else [ins] for ins in instructions], [])

    state = map_.get_initial_state()
    states = [state]
    for instruction in instructions:
        newstate = map_.move(state=state, instruction=instruction)
        if newstate != state:
            state = newstate
            states.append(state)
        #
    return states


def compute_final_password(state: State) -> int:
    """Computes a password from the final state."""

    row, col = state.pos
    facing = state._directions.index(state.dir_)

    res = 1000*(1 + row) + 4*(1 + col) + facing
    return res


def main():
    raw = read_input()
    layout, instructions = parse(raw)

    map_flat = Map(layout=layout)
    path = make_full_path(map_=map_flat, instructions=instructions)
    star1 = compute_final_password(path[-1])
    print(f"The final password is {star1}.")

    map_cube = Cube(layout=layout)
    path2 = make_full_path(map_=map_cube, instructions=instructions)
    star2 = compute_final_password(path2[-1])
    print(f"The final password using the cubic map is {star2}.")


if __name__ == '__main__':
    main()
