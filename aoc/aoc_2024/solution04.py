# +·.`·.· ·`*      .·.` *·  `.`    *+·  · `   *·. `` +. · ·*   ·`·. ·+  `·*  ..·
#  `·. · *`   ·· ·+   . ·`· *` .   Ceres Search  *·  .`  ·*  · `  *·  `• * ·.``·
# .·`·*     .· `·+  .· https://adventofcode.com/2024/day/4 .·   · ·`* .· ` .*··`
# ·.·•` *` ·   + `.· * ·` *· ·.+` · `    · `* · ·` .`· .+  · *`.    `·   .·*` · 


import numpy as np


def parse(s: str):
    res = np.array([list(line) for line in s.splitlines()])
    return res


def find_char(arr, char):
    """Iterates over the indices of an array where the value is the target character"""
    for ind in np.ndindex(arr.shape):
        if arr[ind] == char:
            yield ind
        #
    #


def _add_tuples(a, b):
    res = tuple(x + y for x, y in zip(a, b, strict=True))
    return res


def get_substring(arr, ind: tuple, direction: tuple, n_chars: int):
    """Given an array and index, returns the substring consisting of (up to) the n characters in the specified direction."""
    
    # Get the indices containing the substring
    crds = [ind]
    while len(crds) < n_chars:
        crds.append(_add_tuples(crds[-1], direction))
    
    # Get the characters
    chars = []
    for crd in crds:
        # Stop iterating if we fall off the array
        out_of_bounds = not all(0 <= x < dim for x, dim in zip(crd, arr.shape))
        if out_of_bounds:
            break
        
        chars.append(arr[crd])
    
    res = "".join(chars)
    return res


def count_xmas(arr) -> int:
    """Counts occurrences of 'XMAS' in all 8 spatial directions"""
    
    searchfor = "XMAS"
    n = 0
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    
    # Find all instances of the first character, and iterate in all directions from there
    for ind in find_char(arr=arr, char=searchfor[0]):
        for dir_ in directions:
            # Check for matches
            substring = get_substring(arr=arr, ind=ind, direction=dir_, n_chars=len(searchfor))
            n += substring == searchfor
        #
    
    return n


def count_x_mas(arr) -> int:
    """Counts occurrences of an 'X' of 'MAS' substrings."""
    
    # Search for the middle character in the target string
    searchfor = "MAS"
    assert len(searchfor) % 2 == 1
    offset = len(searchfor) // 2
    
    # There's a 'MAS' X if there's a match in both the NE-SW and NW-SE directions
    diagonals = ((1, 1), (-1, -1))
    antidiagonals = ((1, -1), (-1, 1))
    n = 0
    
    def is_match(ind, dirs):
        """Checks for matches along the input directions"""
        for dir_ in dirs:
            pos = tuple(x - offset*delta for x, delta in zip(ind, dir_, strict=True))
            substring = get_substring(arr=arr, direction=dir_, ind=pos, n_chars=len(searchfor))
            if substring == searchfor:
                return True
            #
        return False
    
    for ind in find_char(arr=arr, char=searchfor[offset]):
        # Found a 'MAS'-X if there's a 'MAS' string along both diagonal directions
        n += all(is_match(ind, dirs) for dirs in (diagonals, antidiagonals))
    
    return n


def solve(data: str) -> tuple[int|str, int|str]:
    parsed = parse(data)
    
    star1 = count_xmas(parsed)
    print(f"Solution to part 1: {star1}")

    star2 = count_x_mas(parsed)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 4
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()