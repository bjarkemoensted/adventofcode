# · *·`.+· ` ·  ·  ·   •· .+ ·  ·`   ·  ·   *·.  `·  · ·.   +`  ·. ·+ `* ·* ·.··
# +·`.·•· ·    ·`. `.*·  ·` ·   Mine Cart Madness   · ` ·   ·*` · . · *`·  `*·.·
# `·  .·   · ·  *`. ·· https://adventofcode.com/2018/day/13  ·    ·.  · · ·+ `· 
# ·.·*·` `  · .  ·`+ ` · + .· · ·*`· · +. · ·*  ·   ·`.    ·  +·.·*    ·.   ··` 

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import cache
from typing import TypeAlias

import numpy as np

# For holding i, j-coordinates
coordtype: TypeAlias = tuple[int, int]

# For inferring which trck types must be under which cart symbols
_infer_track = {
    "^": "|",
    "v": "|",
    "<": "-",
    ">": "-"
}

# Define directions and left/right turns from any direction
drul = [(1, 0), (0, 1), (-1, 0), (0, -1)]
down, right, up, left = drul
dir2ind = {_dir: i for i, _dir in enumerate(drul)}
_rightturns = {drul[i]: drul[(i-1)%len(drul)] for i in range(len(drul))}
_leftturns = {drul[i]: drul[(i+1)%len(drul)] for i in range(len(drul))}


def parse(s: str) -> np.typing.NDArray[np.str_]:
    """Parse input into array of chars"""
    chars = [[char for char in line] for line in s.splitlines()]
    cols = len(chars[0])
    assert all(len(row) == cols for row in chars)
    
    res = np.array(chars, dtype='<U1')
    return res


@cache
def step(pos: coordtype, dir_: coordtype) -> coordtype:
    """Takes a single step in specified direction from given point"""
    res = tuple(a+b for a, b in zip(pos, dir_, strict=True))
    assert len(res) == 2
    return res


# Map arrow symbols to directions and vice versa
directions: dict[str, coordtype] = {
    "^": up,
    "v": down,
    "<": left,
    ">": right
}

directions_inverse = {dir_: char for char, dir_ in directions.items()}

# Define turns when hitting turns in the tracks
turns = {
    '/': {down: left, right: up, up: right, left: down},
    '\\': {down: right, right: down, up: left, left: up}
}


class Cart:
    """Represents a cart. Handles stuff like updating position and direction given current status"""
    
    # Handles turn logic (alternates between left/straight/right) at intersections
    turn_order = [
        _leftturns,
        {dir_: dir_ for dir_ in drul},
        _rightturns
    ]
    
    def __init__(self, pos: coordtype, dir_: coordtype, mine: Mine):
        self.pos = pos
        self.dir_ = dir_
        self.ind = 0  # This determines the direction hoice at next intersection
        self.mine = mine  # reference to the mine for figuring out tracks at current position
    
    def tick(self) -> coordtype:
        """Updates the current state of the cart"""
        
        # Move one step in active direction
        self.pos = step(pos=self.pos, dir_=self.dir_)
        # Determine track at new location and whether to turn
        char = self.mine.ascii_track[*self.pos]
        
        if char == "|" or char == "-":
            pass
        elif char == "+":
            self.dir_ = self.turn_order[self.ind][self.dir_]
            self.ind = (self.ind + 1) % len(self.turn_order)
        else:
            self.dir_ = turns[char][self.dir_]
            
        return self.pos
    #


class Mine:
    def __init__(self, map_: np.typing.NDArray[np.str_]):
        assert len(map_.shape) == 2
        self.shape: coordtype = map_.shape
        
        self.ascii_track = np.full(self.shape, " ", dtype='<U1')
        self.carts: list[Cart] = []
        self.setup(map_=map_)
    
    def setup(self, map_: np.typing.NDArray[np.str_]) -> None:
        """Sets up the mine, given the char array data from the input"""
        for i, j in np.ndindex(self.shape):
            char = map_[i, j]
            if char == " ":
                continue
            
            # Add track to ascii data (to make plots of the current map)
            track_sym = _infer_track.get(char, char)
            self.ascii_track[i, j] = track_sym
            
            # If there's a cart here, add it to the cart list
            is_cart = char in _infer_track
            if is_cart:
                dir_ = directions[char]
                cart = Cart(pos=(i, j), dir_=dir_, mine=self)
                self.carts.append(cart)
            #
        #
    
    def as_string(self) -> str:
        """Represents the current state of the mine as a string.
        This is useful for debugging etc, as it looks like the examples
        from the web page."""
        
        track = self.ascii_track.copy()
        # Replace the track characters with cart symbols
        for cart in self.carts:
            track[*cart.pos] = directions_inverse[cart.dir_]
        
        lines = ("".join(line) for line in track)
        res = "\n".join(lines)
        return res
    
    def tick(self, remove_crashed=False) -> bool:
        """Update all cart states. If remove_crashed, will immediately remove crashed carts and carry on.
        Otherwise, will stop immediately when detecting a crash.
        Returns a bool, denoting whether the tick completed without any collisions."""
        
        # Update carts, going top to bottom and left to right
        order = sorted(range(len(self.carts)), key=lambda i: self.carts[i].pos)
        crashed: set[int] = set([])  # carts collided in this tick
        
        for i in order:
            # Check if any other carts occupy the new position.
            new_pos = self.carts[i].tick()
            share_pos = {ind for ind, cart in enumerate(self.carts) if cart.pos == new_pos and ind not in crashed}
            collision = len(share_pos) > 1
            
            if collision:
                if not remove_crashed:
                    return False
                crashed |= share_pos
            #
        
        # Remove any carts that crashed in this round
        if crashed:
            for i in range(len(self.carts)-1, -1, -1):
                if i in crashed:
                    del self.carts[i]
                #
            #
        return True
    #


def determine_collision(mine: Mine) -> coordtype:
    """Determines the coordinate of the first collision"""
    
    mine = deepcopy(mine)
    # Keep updating until a collision happens
    while mine.tick():
        pass
    
    # Just grab the first position with more than one carts
    counts = Counter((cart.pos for cart in mine.carts))
    collision_coord = next(pos for pos, n in counts.items() if n > 1)
    return collision_coord


def determine_final_cart_loc(mine: Mine):
    """Determine the position of the last remaining cart after all other carts have crashed."""
    mine = deepcopy(mine)
    while len(mine.carts) > 1:
        mine.tick(remove_crashed=True)
    res = mine.carts[0].pos
    return res


def solve(data: str) -> tuple[int|str, int|str]:
    map_ = parse(data)
    mine = Mine(map_)
    
    i, j = determine_collision(mine)
    star1 = f"{j},{i}"
    print(f"Solution to part 1: {star1}")

    i2, j2 = determine_final_cart_loc(mine)
    star2 = f"{j2},{i2}"
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 13
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
