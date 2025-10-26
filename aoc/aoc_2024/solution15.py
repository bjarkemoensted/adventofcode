# . `·•+.   *· ` · . +*`· . ·  ·`  *.`· + `·   ·.`  ·  ·`  .  ·  .· + `.·   *·`·
# *.·`   ·.    .· *  ·.·`• ·      Warehouse Woes     ·  .· `*.•·  · `.· +· .  ·.
# ·.  ··*`  · ·  `   . https://adventofcode.com/2024/day/15 ··   •  · .+ `· . .`
# ··..*`· ·  * ·  ` ·.  +·`* · .·` .   *·   ·*   ·` .+ .·*     `.·* .· `.•`· ·*.


from functools import cache
import numpy as np


dirs = {
    ">": (0, 1),
    "v": (1, 0),
    "^": (-1, 0),
    "<": (0, -1)
}


def parse(s: str):
    map_part, move_part = s.split("\n\n")
    moves = [move for move in move_part.replace("\n", "")]
    
    return map_part, moves


@cache
def add_tuples(a: tuple, b: tuple) -> tuple:
        res = tuple(xa + xb for xa, xb in zip(a, b, strict=True))
        return res


class Warehouse:
    """Represents a Lanternfish warehouse. Keeps track of locations of the robots and crates."""

    robot = "@"
    crate = "O"
    empty = "."
    wall = "#"
    crate_gps_corner_symbol = crate  # The symbol to use for computing locations

    @staticmethod
    def parse_map(s: str) -> np.ndarray:
        m = np.array([list(line.strip()) for line in s.splitlines()])
        return m
    
    def __init__(self, map_ascii: np.ndarray, validate=False):
        """Initializes the warehouse. map_ascii is the ASCII representation of the warehouse from the input.
        validate indicates whether we run sanity checks after every move operation."""

        self.m = self.parse_map(map_ascii)
        self._validate = validate
        
        # Set robot location
        locs = np.argwhere(self.m == "@")
        assert len(locs) == 1
        self.x = tuple(int(elem) for elem in locs[0])
    
    def __str__(self):
        s = "\n".join([''.join(line) for line in self.m])
        return s

    def validate(self):
        if not self.m[*self.x] == self.robot:
            raise RuntimeError(f"Coord error: Robot is not at {self.x}. Correct loc: {np.argwhere(self.m == self.robot)}")
        #
    
    def set_values(self, from_to: dict):
        """Takes a dict where keys and values represent indices for the warehouse.
        For each key pair k, v, the value at k will be moved to v. The k-values are replaced by empty spaces
        if they're not overwritten."""

        # Map target coordinates to the values they should contain after the operation
        d = dict()
        for a, b in from_to.items():
            assert b not in d
            val = self.m[*a]
            
            # Set the coordinate as empty
            self.m[*a] = self.empty
            d[b] = val
        
        # Write the values
        for ind, val in d.items():
            self.m[*ind] = val
            if val == self.robot:
                self.x = ind
            #
        
        # Optional sanity check
        if self._validate:
            self.validate()
    
    def _ind_group(self, ind: tuple):
        """Returns a list of the indices storing the object located at the input index.
        This can be extended for cases where an object takes up more than one index"""
        
        return [ind]

    def move(self, move_char: str):
        """Attempts to move the robot in the input direction."""
        
        dir_ = dirs[move_char]
        from_to = dict()
        
        # Maintain a 'frontier' of the indices where we have yet to figure out what to do
        frontier = self._ind_group(self.x)
        
        while frontier:
            next_ = []
            for ind in frontier:
                # Skip if we already treated this index
                if ind in from_to:
                    continue
                
                # If the target index is new, figure out what to do based what's stored at the index
                target = add_tuples(ind, dir_)
                
                match self.m[*target]:
                    case self.wall:
                        return  # Abort the whole moving operation if anything hits a wall
                    case self.empty:
                        pass  # No issue if the target is free
                    case _:
                        # If there's a crate at the target, we'll need to work those out in the subsequent iteration
                        next_ += self._ind_group(target)
                    #
                
                from_to[ind] = target
            
            frontier = sorted(set(next_))
        
        # We only make it here if everything resolves nicely, so execute the move
        self.set_values(from_to=from_to)

    def execute_moves(self, moves: list):
        for i, dir_ in enumerate(moves):
            self.move(dir_)
        #

    def compute_gps_score(self):
        res = sum(100*i + j for i, j in np.argwhere(self.m == self.crate_gps_corner_symbol))
        return res


class Widehouse(Warehouse):
    crate_left = "["
    crate_right = "]"
    crate = crate_left+crate_right
    crate_gps_corner_symbol = crate_left

    def parse_map(self, s: str) -> np.ndarray:
        """Extend the map as per the riddle"""
        replacements = [("#", "##"), ("O", "[]"), (".", ".."), ("@", "@.")]
        for old, new in replacements:
            s = s.replace(old, new)
        
        return super().parse_map(s)
    
    def validate(self):
        """Also validate that no crates are split in halves"""

        for ind in np.argwhere(self.m == self.crate_left):
            i, j = map(int, ind)
            if self.m[i, j+1] != self.crate_right:
                raise RuntimeError(f"Crate symbol {self.crate_left} at {(i, j)} has no other half!")
            #
        super().validate()
    
    def _ind_group(self, ind: tuple):
        """If there's a crate at the index, return both indices where the crate is located"""
        i, j = ind
        match self.m[*ind]:
            case self.crate_left:
                return [ind, (i, j+1)]
            case self.crate_right:
                return [(i, j-1), ind]
            case _:
                return super()._ind_group(ind=ind)
            #
        #
    #

    
def solve(data: str) -> tuple[int|str, int|str]:
    m, moves = parse(data)

    warehouse = Warehouse(m)
    warehouse.execute_moves(moves)
    
    star1 =  warehouse.compute_gps_score()
    print(f"Solution to part 1: {star1}")
    
    wh2 = Widehouse(m)
    wh2.execute_moves(moves)

    star2 = wh2.compute_gps_score()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 15
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()