# `. ꞏ  ⸳ꞏ* `   . • ⸳ꞏ     `*.ꞏ⸳  `ꞏ.• *`  ꞏ         +ꞏ⸳`  ꞏ .`  *ꞏ`    `*⸳ •ꞏ ꞏ
# *⸳  `    *+⸳ ` •.    +        ` Code Chronicle * ⸳ꞏ.    + *` .⸳  ꞏ`     * ꞏ .+
#  + •⸳`ꞏ  + ꞏ ⸳ . ꞏ + https://adventofcode.com/2024/day/25 `⸳ꞏ*ꞏ⸳`   •⸳  . +ꞏ `
#  `+. ꞏ* ⸳.`  *ꞏ ⸳`    ꞏ.*⸳ `• ꞏ.  +ꞏ`  ⸳   . ꞏ* `ꞏ+    .` ꞏ    ꞏ. ⸳⸳.* `  ⸳  .


import numpy as np


def parse(s):
    locks = []
    keys = []
    
    for part in s.split("\n\n"):
        # Figure out if this part represents a lock or key, then discard the top and bottom rows
        m = np.array([list(line) for line in part.splitlines()])
        is_lock = all(char=="#" for char in m[0])
        pin_area = m[1:-1, :]
        
        # The 'shape' of the key/lock is then the number of "#" in each column
        shape = tuple(sum(char == "#" for char in col) for col in pin_area.T)
        list_ = locks if is_lock else keys
        list_.append(shape)
        
    return locks, keys


def count_fitting_key_lock_paris(locks: list, keys: list, pinsize=5):
    """Takes lists of lock and key shapes. Returns the number of compatible pairs, i.e.
    pairs where there's room for the key inside the lock."""
    
    res = 0
    for lock in locks:
        # Determine the max pin heights for a key to fit in this lock
        max_pin_heights = tuple(pinsize - pin_height for pin_height in lock)
        for key in keys:
            fits = not any(pin > max_ for pin, max_ in zip(key, max_pin_heights))
            res += fits
        #

    return res


def solve(data: str):
    locks, keys = parse(data)

    star1 = count_fitting_key_lock_paris(locks=locks, keys=keys)
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2024, 25
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
