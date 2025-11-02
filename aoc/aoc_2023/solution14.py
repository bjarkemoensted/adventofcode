# +·· `*.·     ·+·  ` *· ·. ` + · .·  *`  · •.·  `· +· · *·  . ·  ·  *`  ···.* ·
# · `*.· .*·  ·  +`· .`•  ·· Parabolic Reflector Dish · .   `·   *  .·  ·+  ·`.•
# ··.`+`· .*  .   ·` • https://adventofcode.com/2023/day/14 ·*·`    ·  •· .` ·· 
# ·.* · `+  ··   . * ·· `+     · .`·•.. * ·    `*·` .    ··•    `· ·*.··` · *.· 

from __future__ import annotations
from functools import cache
import numpy as np
from numpy.typing import NDArray
from typing import Iterable, TypeAlias


coordtype: TypeAlias = tuple[int, int]


_rock = "O"
_free = "."
_block = "#"


def parse(s: str) -> dict[coordtype, str]:
    res: dict[coordtype, str] = dict()
    for i, line in enumerate(s.splitlines()):
        for j, symbol in enumerate(line):
            res[(i, j)] = symbol
        #

    return res


north = (-1, 0)
west = (0, -1)
east = (0, 1)
south = (1, 0)

all_dirs = (north, west, south, east)


@cache
def dir_as_arr(dir_: coordtype) -> NDArray[np.int_]:
    res = np.array(dir_)
    return res


class Environment:
    """Handles environmental things, i.e. objects that are independent of the rocks that roll around.
    This is to separate the 'fixed' objects which remain constant even when tilting the platform, from
    the rocks whose positions will change frequently.
    The underlying reasoning is to have a concise representation of the state of the platform,
    without many redundant copies of things like block locations, which are shared between
    different states."""

    def __init__(self, shape: coordtype, blocks: Iterable[coordtype]) -> None:
        self.shape = shape
        self.blocks = set(blocks)
    
    @cache
    def position_is_valid(self, pos: coordtype) -> bool:
        """Determines whether a rock position is valid, given the environment"""
        if not all(0 <= x < lim for x, lim in zip(pos, self.shape, strict=True)):
            return False
        
        return pos not in self.blocks
    
    @cache
    def slide_coords(self, pos: coordtype, dir_: coordtype) -> tuple[coordtype, ...]:
        """Returns a tuple of all the coordinates in the environment which can be reached
        by starting at the input position, then sliding in the provided direction, until hitting
        a block, or going out of bounds."""

        running = pos
        res_list = []
        di, dj = dir_
        
        while self.position_is_valid(running):
            i, j = running
            res_list.append((i, j))
            running = (i+di, j+dj)
        
        res = tuple(res_list)
        if not res:
            raise ValueError(f"Invalid position: {pos}")
        
        return res


class Platform:
    def __init__(self, env: Environment, rocks: Iterable[coordtype]) -> None:
        self.env = env
        self.rocks = frozenset(rocks)

    @classmethod
    def from_layout(cls, layout: dict[coordtype, str]) -> Platform:
        """Instantiate a platform (including its environment) from a layout dict"""
        
        shape = tuple(max(vals) + 1 for vals in zip(*layout.keys()))
        block_locs = (pos for pos, sym in layout.items() if sym == _block)
        env = Environment(shape=shape, blocks=block_locs)
        
        rock_locs = ((pos for pos, sym in layout.items() if sym == _rock))
        res = cls(env=env, rocks=rock_locs)
        return res

    def as_array(self) -> NDArray[np.str_]:
        """Represents the state as a numpy array"""
        m = np.full(self.env.shape, _free, dtype=np.str_)
        categories = (
            (self.env.blocks, _block),
            (self.rocks, _rock)
        )
        
        for locs, sym in categories:
            for i, j in locs:
                m[i, j] = sym
            #
        
        return m

    def as_string(self) -> str:
        m = self.as_array()
        res = "\n".join(("".join(row) for row in m))
        return res
    
    def get_rock_inds(self, dir_: coordtype) -> list[coordtype]:
        """Returns rock indices indices, for computing the locations of rocks resulting from tilting
        the platform in the specified direction.
        Indices are generated e.g. north-to-south if tilting north, etc, to avoid issues where
        rocks prevent each other from rolling."""
        
        # Default order is N->S, W->E. Reverse this if tilting to the south or east
        reverse = dir_ in (south, east)
        res = sorted(self.rocks, reverse=reverse)
        
        return res

    def tilt(self, dir_: coordtype) -> Platform:
        """Tilts the platform, updating rock positions. Returns a new platform instance,
        with the updated rock positions."""
        
        new_rocks: set[coordtype] = set([])
        
        for pos in self.get_rock_inds(dir_):
            path = self.env.slide_coords(pos=pos, dir_=dir_)
            new_pos = next(x for x in reversed(path) if x not in new_rocks or x == pos)
            
            new_rocks.add(new_pos)
            
        res = Platform(env=self.env, rocks=new_rocks)
        return res
    
    def calculate_load(self) -> int:
        offset, _ = self.env.shape
        res = sum(offset - i for i, _ in self.rocks)
        return res
    #


def simulate_cycles(initial_state: Platform, n=1_000_000_000) -> Platform:
    running = initial_state
    history: dict[frozenset, int] = dict()
    seen_states: list[Platform] = []
    
    for i in range(n):
        key = running.rocks
        
        try:
            # Look for recurrences so far
            loop_start_ind = history[key]
            
            # Compute the index of the final state
            loop_len = i - loop_start_ind
            n_cycles_left = n - i
            final_ind = loop_start_ind + (n_cycles_left % loop_len)
            
            final_state = seen_states[final_ind]
            return final_state
        except KeyError:
            # If no recurrence yet, add the current state to the history
            history[key] = i
            seen_states.append(running)
            
        for dir_ in all_dirs:
            running = running.tilt(dir_)
        #

    return running


def solve(data: str) -> tuple[int|str, ...]:
    layout = parse(data)
    
    platform = Platform.from_layout(layout)
    
    star1 = platform.tilt(north).calculate_load()
    print(f"Solution to part 1: {star1}")

    platform_final = simulate_cycles(initial_state=platform)
        
    star2 = platform_final.calculate_load()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 14
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
