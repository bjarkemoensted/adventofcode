# ⸳+ꞏ   .` ⸳• ꞏ.*  *   •ꞏ .  ⸳ `ꞏ. *`   ꞏ    ꞏ.* `.*ꞏ⸳ . `• *   ⸳``  +ꞏ⸳.+    ꞏ⸳
#  .**  ꞏ. ` ⸳   * .+ꞏ`⸳   `•ꞏ + Chocolate Charts  ⸳`  ꞏ  `* `.+   ⸳   ꞏ .•*`ꞏ.•
# *ꞏ⸳  ``⸳ +⸳ꞏ  ⸳  `.  https://adventofcode.com/2018/day/14  ⸳ꞏ`+*     .  ⸳⸳*  `
# ꞏ⸳. `. + ⸳  • ` ⸳  +⸳.   *ꞏ   .  `ꞏ`+ ⸳ ⸳ꞏ* `   *ꞏ.  ⸳ .⸳*    `. *⸳`⸳ ꞏ .`⸳ *ꞏ

import numba
import numpy as np

# Data type for representing scores
score_dtype = np.int8

# Sentinel value to indicate values not set yet
_sentinel = -1

# The max number of new recipes per step in the algorithm
MAX_STEP_SIZE = 2


@numba.njit
def brew(elf1: int, elf2: int, scores: np.typing.NDArray[score_dtype], n_add: int) -> tuple[int, int, np.typing.NDArray[score_dtype]]:
    """Given the current recipes of the two elves, and the current list of scores, adds another n recipes"""
    
    # Allocate a larger array for results
    res_size = len(scores) + n_add
    res = np.full(res_size, _sentinel, dtype=score_dtype)
    res[:len(scores)] = scores
    
    # find effective array length (first sentinel value)
    edge = len(scores)
    assert res[edge] == _sentinel
    # Scan backwards in case we have some leftover sentinel values
    while edge > 0 and res[edge-1] == _sentinel:
        edge -= 1  # There's a sentinel value to the left, so move edge one step left
    
    # Keep adding recipes until we've almost reached the desired number (going over will cause index errors)
    
    while edge < res_size - MAX_STEP_SIZE:
        combi = res[elf1] + res[elf2]
        # Update recipes and edge location
        if combi >= 10:
            res[edge] = combi // 10
            edge += 1
        res[edge] = combi % 10
        edge += 1
        
        # Update the elves' current recipes (the .item thing is to avoid casting to few-bits ints and overflowing)
        elf1 = (elf1 + 1 + res[elf1].item()) % edge
        elf2 = (elf2 + 1 + res[elf2].item()) % edge
        
    return elf1, elf2, res


def brew_until_sequence(
        elf1: int,
        elf2: int,
        scores: np.typing.NDArray[np.int_],
        sequence: list[int],
        batch_size: int=5_000_000) -> int:
    """Repeatedly brews 'batches' of new hot chocolate recipes of the specified batch size, until
    the specified sequence of scores appears, at which point the index where the sequence appears is returned"""
    
    left_limit = 0  # keeps track of how far into the list of scores we've checked for the sequence
    seq = np.array(sequence, dtype=score_dtype)
    
    while True:
        # Check if the sequence has appears
        windows = np.lib.stride_tricks.sliding_window_view(scores[left_limit:], seq.size)
        
        n = len(scores[left_limit:]) - seq.size + 1
        assert n == windows.shape[0], (n, windows.shape[0])
        
        matches = np.all(windows == seq, axis=1)
        match_idx = np.where(matches)[0]
        if match_idx.size > 0:
            return left_limit + match_idx[0]
        
        # Update the left limit, so we don't have to recheck unnecessarily
        left_limit = max(0, len(scores) - len(sequence) - 3)
        
        # Brew another batch of recipes
        elf1, elf2, scores = brew(elf1, elf2, scores, n_add=batch_size)
    #


def solve(data: str):
    n_recipes = int(data)
    scores = np.array([3, 7], dtype=score_dtype)
    elf1, elf2 = 0, 1
    
    # Brew enough recipes so we can check the 10 scores after the n'th recipe
    n_chars = 10
    # Add a small 'buffer' number of recipes, because the algorithm might stop a few steps short
    elf1, elf2, scores = brew(elf1, elf2, scores, n_add=n_recipes + n_chars + MAX_STEP_SIZE)
    
    star1 = "".join(map(str, scores[n_recipes: n_recipes+n_chars]))
    print(f"Solution to part 1: {star1}")
    
    # Keep breweing batches until we hit the target sequence of scores
    sequence = [int(char) for char in data]
    star2 = brew_until_sequence(elf1, elf2, scores, sequence)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2018, 14
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
