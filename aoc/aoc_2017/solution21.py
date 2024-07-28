import numpy as np


def get_initial():
    """Make ths starting state"""
    res = np.array([list('.#.'), list('..#'), list('###')])
    return res


def _numpy_to_tuple(arr):
    res = tuple(''.join(row) for row in arr)
    return res


def _tuple_to_numpy(tup: tuple):
    res = np.array([list(chars) for chars in tup])
    return res


def parse(s):
    """Make a hashable representation of each array to the outputs"""
    res = dict()
    for line in s.splitlines():
        a, b = [np.array([list(chars) for chars in part.split("/")]) for part in line.split(" => ")]
        k = _numpy_to_tuple(a)
        v = np.array([list(chars) for chars in b])
        res[k] = v

    return res


def _iterate_subpics(state, partsize: int):
    """Iterate over the smaller subsections of ana rray"""
    rows, cols = state.shape
    for ni, i in enumerate(range(0, rows, partsize)):
        for nj, j in enumerate(range(0, cols, partsize)):
            subpic = state[i: i+partsize, j: j+partsize]
            yield (ni, nj), subpic


def determine_partsize(state):
    rows, cols = state.shape
    assert rows == cols
    size = rows
    for n in (2, 3):
        if size % n == 0:
            return n
        #
    raise ValueError


def _iterate_flip_rotate(m: np.array):
    """Iterates over every rotation/flip of an array"""
    flipped = [np.flip(m.copy(), axis=i).copy() for i in (0, 1)]

    for arr in [m]+flipped:
        for k in range(4):
            res = np.rot90(arr, k=k).copy()
            yield res


def make_lookup(rules: dict):
    """Make lookup dict of all possible reorientations"""
    res = dict()

    for k, v in rules.items():
        key_arr = _tuple_to_numpy(k)
        for equivalent_pic in _iterate_flip_rotate(key_arr):
            newkey = _numpy_to_tuple(equivalent_pic)
            if newkey in res:
                if not all((v == res[newkey]).flat):
                    raise ValueError
                #
            else:
                res[newkey] = v
            #
        #
    return res


class Matcher:
    def __init__(self, rules: dict):
        self.cached = make_lookup(rules)

    def __call__(self, pic):
        key = _numpy_to_tuple(pic)
        res = self.cached[key]
        return res


def single(state, matcher: Matcher):
    partsize = determine_partsize(state)
    size, _ = state.shape

    n_patterns = size // partsize
    newsize = size + n_patterns

    unfilled = "x"
    res = np.array([[unfilled for _ in range(newsize)] for _ in range(newsize)])
    for (ni, nj), subpic in _iterate_subpics(state, partsize):
        replacement = matcher(subpic)
        replacement_size, _ = replacement.shape

        if not all(char == unfilled for char in res[ni*newsize: (ni+1)*newsize, nj*newsize: (nj+1)*newsize].flat):
            raise ValueError("About to fill values that are already filled, so something's wrong :/")

        i1 = ni * replacement_size
        i2 = (ni + 1) * replacement_size

        j1 = nj * replacement_size
        j2 = (nj + 1) * replacement_size
        res[i1: i2, j1: j2] = replacement

    return res


def simulate_growth(rules: dict, n_steps: int = 5, verbose=False, state=None):
    matcher = Matcher(rules)

    if state is None:
        state = get_initial()

    for _ in range(n_steps):
        state = single(state, matcher)
        if verbose:
            print(state)
            print()

    return state


def solve(data: str, n_steps=5):
    rules = parse(data)
    endstate = simulate_growth(rules=rules, n_steps=n_steps)

    star1 = sum(char == "#" for char in endstate.flat)
    print(f"Solution to part 1: {star1}")

    n_final = 18
    n_left = n_final - n_steps
    if n_steps <= 3:
        star2 = None
    else:
        final_state = simulate_growth(rules=rules, n_steps=n_left, state=endstate)
        star2 = sum(char == "#" for char in final_state.flat)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 21
    from aoc.utils.data import check_examples

    parser = lambda s: dict(n_steps=int(s.split("iterations=")[-1]))
    check_examples(year=year, day=day, solver=solve, extra_kwargs_parser=parser)

    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
