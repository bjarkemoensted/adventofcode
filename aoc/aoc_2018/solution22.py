# ··.  `··   *+`·  .  ·•`·.`  · *     · ` ·  .*` ·   ·*.   •·` ·  .  ·  ` *·  ·.
# .`· *.·• ` ·   ·+ .·`  .*  ·  . ` Mode Maze .+   ·.*` ·   .· `.·•`  ·  ·. `·*·
# *· `·   `.·. ·  ·    https://adventofcode.com/2018/day/22 +  . *`·.     `*·.`*
# ·*`·. ` ·   `. . ·` +·. ·  * .`· `   .·+    ·.  *  ·.`  `   . · ·*  .· .  ·*.·

import math
from enum import IntEnum
from heapq import heappop, heappush
from typing import TypeAlias

coord: TypeAlias = tuple[int, int]
# Represents a state in the cave graph (current gear index, coordinates)
statetype: TypeAlias = tuple[int, coord]


class Equipment(IntEnum):
    """The types of equipment that may be carried in the cave"""
    TORCH = 0
    CLIMBING_GEAR = 1
    NEITHER = 2


class Region(IntEnum):
    """The region types encountered in the cave"""
    ROCKY = 0
    WET = 1
    NARROW = 2


# Define the valid equipment/region combos
compatible_gear: dict[int, set[int]] = {
    Region.ROCKY.value: {Equipment.CLIMBING_GEAR.value, Equipment.TORCH.value},
    Region.WET.value: {Equipment.CLIMBING_GEAR.value, Equipment.NEITHER.value},
    Region.NARROW.value: {Equipment.TORCH.value, Equipment.NEITHER.value},
}

# For each region, map each piece of valid equipment to the other valid equipment
valid_switches = dict()
for region_type in Region:
    d_ = dict()
    for e1 in Equipment:
        d_[e1.value] = next(e2.value for e2 in Equipment if e2 != e1 and e2 in compatible_gear[region_type])
    valid_switches[region_type.value] = d_


def parse(s: str) -> tuple[int, coord]:
    a, b = s.splitlines()
    depth = int(a.split("depth: ")[-1])
    target = tuple(int(elem) for elem in b.split("target: ")[-1].split(","))
    assert len(target) == 2
    return depth, target


class Cave:
    """Represents the cave, with helper methods for determining region type, finding optimal
    ways to get to a given point, etc."""
    
    # Components for the formulae for geological index, erosion levels etc
    x_factor = 16807
    y_factor = 48271
    erosion_mod = 20183
    
    # The time required to change equipemnt and travel between adjacent sites
    _gear_change_time = 7
    _travel_time = 1
    origin = (0, 0)
    
    def __init__(self, depth: int, target: coord):
        self.depth = depth
        self.target = target
        self.ncols, self.nrows = self.target
        
        # Set up caches for storing computed values
        self._erosion_level_cache: dict[coord, int] = dict()
        # Formulas give some very large ints, but we only care about moduli, so find the relevant lcm
        self.lcm = math.lcm(self.erosion_mod, len(Region))
    
    def compute_geo_index(self, x: int, y: int) -> int:
        """Computes the 'geological index' of the specified x/y coordinates"""
        
        # Else compute it according to the provided formula
        if (x, y) in (self.origin, self.target):
            res = 0
        elif x == 0:
            res = y*self.y_factor
        elif y == 0:
            res = x*self.x_factor
        else:
            a = self.compute_erosion_level(x-1, y)
            b = self.compute_erosion_level(x, y-1)
            res = a*b
        
        return res
    
    def compute_erosion_level(self, x, y):
        """Compute the erosion level at the specified x/y coordinates"""
        
        # Check for cached result
        key = (x, y)
        if key in self._erosion_level_cache:
            return self._erosion_level_cache[key]
        
        # Compute geo index and use it to compute erosion level
        geo_ind = self.compute_geo_index(x, y)
        res = (geo_ind + self.depth) % self.erosion_mod
        res %= self.lcm
        
        # Add to cache
        self._erosion_level_cache[key] = res
        
        return res
    
    def region_index(self, x: int, y: int) -> int:
        """Compute the region index (index of the region type) at the input coordinates"""
        ind = self.compute_erosion_level(x, y) % len(Region)
        return ind
    
    def as_string(self, ascii_symbols: bool=True) -> str:
        """Represents the cave as a string. For debugging etc.
        if ascii_symbols is True, displays the cave using ASCII art, similar to on the web page."""
        
        symbols = (".", "=", "|") if ascii_symbols else tuple(map(str, range(len(Region))))
        inds = [[self.region_index(x, y) for x in range(self.ncols)] for y in range(self.nrows)]
        res = "\n".join("".join([symbols[ind] for ind in row]) for row in inds)
        return res
    
    @property
    def total_risk_level(self) -> int:
        """Computes the total risk level (sum over region indices in the smallest square enclosing the
        origin and target) of the cave"""
        coords = ((x, y) for x in range(self.ncols+1) for y in range(self.nrows+1))
        res = sum(self.compute_erosion_level(x, y) % len(Region) for x, y in coords)
        return res

    def adjacent_states(self, state: statetype) -> list[tuple[int, statetype]]:
        """Iterate over tuples of (distance, next state) for the allowed states that might
        follow the input one."""
        
        # Generate states by traveling to neighboring sites, keeping current gear
        current_gear, (x, y) = state
        near = ((x+1, y), (x-1, y), (x, y+1), (x, y-1))
        res = [(self._travel_time, (current_gear, pos)) for pos in near]
        
        # Remove candidate states that are physically out of bounds, or involve invalid gear/region combo
        for i in range(len(res)-1, -1, -1):
            _, (cgear, (cx, cy)) = res[i]
            if cx < 0 or cy < 0:
                del res[i]
            else:
                region = self.region_index(cx, cy)
                if cgear not in compatible_gear[region]:
                    del res[i]
            #
        
        # Add option of switching from current gear to the other allowed gear for the region
        res += [(self._gear_change_time, (valid_switches[self.region_index(x, y)][current_gear], (x, y)))]
        
        return res

    def shortest_path(self) -> int:
        """Computes the most efficient way of getting to the target, using A*."""
        
        start = self.origin
        target = self.target
        target_gear = Equipment.TORCH.value
        # We must start and end with the torch equipped
        target_state = (target_gear, target)
        initial_state = (target_gear, start)
        
        # Keep running tally of distances and parents
        d_g = {initial_state: 0}
        came_from: dict[statetype, statetype] = dict()
        
        def heuristic(state_: statetype) -> int:
            """Heuristic lower bound on remaining time (Manhatten dist plus gear change)"""
            gear_, pos_ = state_
            min_travel = self._travel_time*sum(abs(a-b) for a, b in zip(pos_, target))
            min_gear = self._gear_change_time*(gear_ != target_gear)
            return min_travel + min_gear
        
        queue = [(heuristic(initial_state), initial_state)]
        
        while queue:
            _, state = heappop(queue)
            # Return the distance (minutes) if we've reached the target
            if state == target_state:
                return d_g[state]
            
            # Otherwise iterate over neighbors, and keep promising candidates
            for delta, neighbor in self.adjacent_states(state):
                g = d_g[state] + delta
                if g < d_g.get(neighbor, float("inf")):
                    came_from[neighbor] = state
                    d_g[neighbor] = g
                    priority = g + heuristic(neighbor)
                    heappush(queue, (priority, neighbor))
                #
            
        #
        raise RuntimeError


def solve(data: str) -> tuple[int|str, int|str]:
    depth, target = parse(data)
    cave = Cave(depth=depth, target=target)
    
    star1 = cave.total_risk_level
    print(f"Solution to part 1: {star1}")

    star2 = cave.shortest_path()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 22
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
