#   .    .• ⸳  . •⸳`ꞏ  *  ⸳`ꞏ ꞏ • .ꞏ    +`  `.*•ꞏ⸳⸳ +. ꞏ  ⸳ `  ꞏ•  *ꞏ  .`*⸳    .
# `+⸳ꞏ   *.    ` ꞏꞏ. ` ⸳  + Settlers of The North Pole +  +⸳ .ꞏ⸳ .          ꞏ ⸳*
# *`*.     ⸳  `•⸳.•  . https://adventofcode.com/2018/day/18 .+⸳   .ꞏ     `ꞏ+ `. 
# *.ꞏ`ꞏ   * . ⸳ꞏ`   .  ꞏ ⸳ ⸳+ꞏ   . `• ⸳ ꞏꞏ+. `    ꞏ•  .   ꞏ.⸳ +ꞏ   ⸳` ꞏꞏ+ • ` ꞏ⸳

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import TypeAlias

arrtype: TypeAlias = np.typing.NDArray[np.str_]


class symbols:
    open_ground = "."
    tree = "|"
    lumberyard = "#"

_symbols = (symbols.open_ground, symbols.tree, symbols.lumberyard)


class Woodland:
    """Represents the woodlands and exposes methods for state computations"""
    
    def __init__(self, map_: arrtype, neighborhood_size: int = 3):
        # Store the area of the neighborhoods to scan, and the amount of padding needed for edges
        self.adj = (neighborhood_size, neighborhood_size)
        self.padsize = neighborhood_size // 2
        # Cache arrays fully populated with each symbol, for comparing against sliding windows in data
        self.filled = {char: np.full(shape=map_.shape, fill_value=char, dtype='<U1') for char in _symbols}

        # If a recurrence is found, stores the first occurrence time
        self.recurrence: int|None = None
        self.hashes: dict[int, int] = dict()
        
        # Record the initial state
        self.states: list[arrtype] = []
        self._record_state(map_.copy())
    
    def _record_state(self, state: arrtype) -> None:
        """Records a given state, i.e. saves its hash value and checks if a recurrence occurred."""
        
        # Recurrence math relies on the length of the states list, so make sure we stop recording
        assert self.recurrence is None
        hash_ = hash(state.tobytes())
        
        try:
            # Check if state is seen before
            n_last_seen = self.hashes[hash_]
            self.recurrence = n_last_seen
        except KeyError:
            # If not, add its hash
            self.hashes[hash_] = len(self.states)
            self.states.append(state)
        
    def _get_counts(self, m: arrtype, char: str) -> np.typing.NDArray[np.int_]:
        """Compares the input array against the input character.
        Returns an array of the number of matches in the neighborhood around each cell,
        excluding the cell itself."""
        
        # Compare against an array full of the specified char. Pad with False to include edges.
        mask = (m == self.filled[char])
        padded = np.pad(mask, pad_width=self.padsize, mode='constant', constant_values=False)
        windows = sliding_window_view(padded, self.adj)
        # Count matches in the (3x3) neighborhood around each cell. Subtract matches from self
        counts = windows.sum(axis=(2, 3)) - mask
        return counts
        
    def tick(self) -> None:
        """Computes one more state"""
        
        m = self.states[-1]
        res = m.copy()
        # Open ground and trees update based solely on number of trees/lumber yards in their surroundings
        res[np.where((m == symbols.open_ground) & (self._get_counts(m, symbols.tree) >= 3))] = symbols.tree
        res[np.where((m == symbols.tree) & (self._get_counts(m, symbols.lumberyard) >= 3))] = symbols.lumberyard
        
        # Lumberyards are different. To remain, they need both lumberyards and trees around them.
        # Update inds are found by negating that using De Morgan: ~(a & b) = ~a | ~b 
        inds = np.where(
            (m == symbols.lumberyard) & (
                (self._get_counts(m, symbols.lumberyard) < 1) |
                (self._get_counts(m, symbols.tree) < 1)
            )
        )
        res[inds] = symbols.open_ground
        
        self._record_state(res)
    
    def get_state_after_n_rounds(self, n: int) -> arrtype:
        """Determine the state after n iterations"""
        
        # Compute new state until we can figure out the state after n its
        while len(self.states) - 1 < n and self.recurrence is None:
            self.tick()
        
        if len(self.states) - 1 >= n:
            ind = n  # lookup if we computed all n states
        elif self.recurrence is not None:
            # If reccurrence, skip cycles and look up the end state
            cycle_length = len(self.states) - self.recurrence
            remainder = (n - self.recurrence) % cycle_length
            ind = self.recurrence + remainder
        else:
            raise RuntimeError
        
        res = self.states[ind]
        return res
    
    def compute_resource_value(self, after_n_steps: int) -> int:
        """Computes the 'resource value' of the state after n iterations"""
        m = self.get_state_after_n_rounds(after_n_steps)
        res = np.equal(m, symbols.tree).sum() * np.equal(m, symbols.lumberyard).sum()
        return res
    
def parse(s) -> arrtype:
    res = np.array([list(line) for line in s.splitlines()], dtype="<U1")
    return res


def solve(data: str):
    map_ = parse(data)
    woodland = Woodland(map_)
    
    star1 = woodland.compute_resource_value(10)
    print(f"Solution to part 1: {star1}")

    star2 = woodland.compute_resource_value(1_000_000_000)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2018, 18
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
