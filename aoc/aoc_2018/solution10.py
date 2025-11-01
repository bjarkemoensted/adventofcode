# · `.`·*.+  .`·*`·.      `·   *.    `·   • *.`+·  `   * . ·•·.  *   . ·`* .  .`
# +· `.*     ·•. *`· `.   + * .  The Stars Align *    .`  · . *  .· `   .· *`   
# ·`  ·.   •` ·  ·     https://adventofcode.com/2018/day/10 `*`*  .· · +  . *. ·
# . · *`   ·    . •`· `*.· *.   ·` .    • · .·  * . • * · `* `·. ` *   .   ·.`·*

import re
from typing import TypeAlias

import aococr
import numpy as np

statetype : TypeAlias = tuple[tuple[int, int], tuple[int, int]]


def parse(s: str) -> list[statetype]:
    states = []
    pattern = re.compile(r"position=<\s*(-?\d+),\s*(-?\d+)>\s+velocity=<\s*(-?\d+),\s*(-?\d+)>")
    
    for line in s.splitlines():
        
        m = re.match(pattern, line)
        assert m is not None
        x, y, vx, vy = map(int, m.groups())
        state = ((x, y), (vx, vy))
        states.append(state)
    
    return states


def decipher(states: list[statetype]) -> tuple[int, str]:
    """Deciphers the message that will appear in the stars. Returns the number of seconds
    required for the message to appear, and the text content."""
    
    # Make numpy arrays for position and velocity
    pos_comp, v_comp = zip(*states)
    pos = np.array([[x, y] for x, y in pos_comp])
    v = np.array([[vx, vy] for vx, vy in v_comp])
    
    n = 0
    
    while True:
        n += 1
        pos += v

        min_ = pos.min(axis=0)
        max_ = pos.max(axis=0)
        xspan, yspan = max_ - min_
        
        # If the stars are spread across a y-range larger than the font height, skip to next second
        clearly_wrong = yspan > 12
        if clearly_wrong:
            continue
        
        # Make a numpy array with ASCII art-type strings, representing (hopefully) the message
        grid = np.full((yspan+1, xspan+1), '.', dtype='<U1')
        for j, i in pos-min_:
            grid[i, j] = "#"
        
        # Attempt to parse it. If success, return time and message
        text = aococr.aococr(grid, fontsize=(10, 6))
        if text:
            return n, text
        #
    #


def solve(data: str) -> tuple[int|str, int|str]:
    initial_states = parse(data)
    n_seconds, message = decipher(initial_states)
    star1 = message
    print(f"Solution to part 1: {star1}")

    star2 = n_seconds
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 10
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
