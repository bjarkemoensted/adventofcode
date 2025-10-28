# · `•·    .·* ·+. `· * `·.·     .  `·       · *    · `.*·  · ·  +   `. *·· .` ·
# ··+ ··   * `  ·· *       . ·  Reservoir Research  *`· `  ·+ .·• `   · ·.* · `.
# .+··`  *.· ·  `*·. · https://adventofcode.com/2018/day/17 ` *  ·. · `·  . •··`
# `.· +`··  * ·       · · ·  * `.·`    •· ·.`   + *· . · *` ·  .·` ·*+ .  · `  ·

from __future__ import annotations
from collections import defaultdict
import functools
import itertools
from typing import Iterable, TypeAlias

coord: TypeAlias = tuple[int, ...]


class symbols:
    clay = "#"
    spring = "+"
    sand = " "
    water = "~"
    passed = "|"


down, right, up, left = ((0, 1), (1, 0), (0, -1), (-1, 0))


@functools.cache
def inc(x: coord, dx: coord) -> coord:
    res = tuple(a+v for a, v in zip(x, dx, strict=True))
    return res


def parse(s: str) -> set[coord]:
    res = set()
    def determine_range(s: str):
        if ".." in s:
            a, b = map(int, s.split(".."))
            return range(a, b+1)
        a = int(s)
        return range(a, a+1)
    
    for line in s.splitlines():
        res |= set(itertools.product(*map(determine_range, (s.split("=")[1] for s in sorted(line.split(", "))))))
    
    return res


class Reservoir:
    """Represents the underground with deposits of water etc"""

    def __init__(self, clay: Iterable[coord], spring_location: coord=(500, 0)):
        # Fix min/max y values for input data because the correct answer only involves points in this yrange
        self._ymin, self._ymax = (sorted([y for _, y in clay])[i] for i in (0, -1))

        # Add clay veins and the water spring
        self.stuff: dict[coord, str] = defaultdict(lambda: symbols.sand)
        self.stuff.update(((c, symbols.clay) for c in clay))
        self.stuff[spring_location] = symbols.spring
        self._spring_location = spring_location

        # Set the y bounds (used to determine if a water stream is out of bounds)
        _, self.y_bounds = ((vals[0], vals[-1]+1) for vals in map(sorted, zip(*self.stuff.keys())))

    def count_water(self, valid_symbols: Iterable[str]|None=None) -> int:
        """Count the number of cells which contain water (defined as one of the input characters,
        defaulting to resting + passed)"""
        if valid_symbols is None:
            valid_symbols = (symbols.water, symbols.passed)
        return sum(char in valid_symbols and self._ymin <= y <= self._ymax for (_, y), char in self.stuff.items())

    def as_string(self, y_range: Iterable[int]|None=None):
        """Represents the underground as an ASCII map.
        y_range can optionally specify a range of y-values to display.
        If no range is provided, displays all y-values."""
       
        # Set default x/y ranges to the entire range of the data
        xr, yr = (list(arr[i] for i in (0, -1)) for arr in map(sorted, zip(*self.stuff.keys())))
        lines = []
        y_range = range(*yr) if y_range is None else y_range

        for y in y_range:
            line = [self[(x, y)] for x in range(*xr)]
            lines.append("".join(line))
        
        res = "\n".join(lines)
        return res

    def __getitem__(self, key: coord):
        return self.stuff[key]
    
    def __setitem__(self, key: coord, val: str):
        self.stuff[key] = val

    def fill(self, n_iterations: int|None=None, interactive_mode=False, display_steps: bool=False):
        """Simulates water pouring from the spring, dripping through sand and fillling any container it encounters.
        n_iterations optionally sets a max number of iterations.
        interactive mode runs a single iteration at a time, proceeding only when a key is pressed.
        display_steps prints the current state at every step."""
        

        # Create a stream of water originating from the spring
        stream = Stream(pos=self._spring_location, reservoir=self)
        
        n_max = float("inf") if n_iterations is None else n_iterations
        n = 0

        while n < n_max:
            # Update the water stream. If nothing happens, we're done
            updated = stream.tick()
            if not updated:
                break
            
            n += 1

            # If necessary, print the current ASCII map
            if interactive_mode or display_steps:
                y_max = stream.get_deepest_y()
                window_size_chars = 120
                eps = 5
                lower = max(0, y_max+eps-window_size_chars)
                _, y_bound = self.y_bounds
                y_range = range(lower, min(y_bound, lower+window_size_chars))
                print(self.as_string(y_range=y_range))
            if interactive_mode:
                # If interactive mode, await next keypress
                input("")

            #
        #
    #


class Stream:
    """Represents a stream of water. This handles the logic determining whether a stream of water drops further down,
    spreads across a horizontal surfaces that it hits, etc.
    When a stream falls off one or more edges, it spawns child streams, which terminate their progress before
    their parent continues."""

    def __init__(self, reservoir: Reservoir, pos: coord):
        self.reservoir = reservoir
        self.ylim = self.reservoir.y_bounds
        
        self.children: list[Stream] = []
        # Keep a stack of positions visited, so if we hit water, we can backtrace
        self._positions = [pos]

        # Keeps track of the highest y-value returned (so as_string doesn't scan back up)
        self._y_display_highest_returned = float("-inf")

    @property
    def pos(self):
        """The current position"""
        return self._positions[-1]

    @pos.setter
    def pos(self, c: coord):
        """Sets the current position"""
        self._positions.append(c)

    def backtrack_in_water(self):
        """Keep backing up until we're out of water"""
        while self.reservoir[self.pos] == symbols.water:
            self._positions.pop()


    def _spawn_child(self, pos: coord) -> None:
        child = self.__class__(pos=pos, reservoir=self.reservoir)
        self.children.append(child)

    def fill_up(self):
        """When we hit a horizontal surface, spread out left and right until we hit a wall or a hole to fall into.
        If holes are encountered, 'child streams' are spawned to progress down the holes before proceeding.
        If we hit walls in bpoth directions, start filling up the current area with water."""

        # Start by iteration in both left/right dirs checking for walls + holes
        points = [self.pos]
        new_fall_coords = []
        confined = True  # this indicates whether we hit walls in both directions
        
        for dir_ in left, right:
            running = inc(self.pos, dir_)#
            # As long as we don't hit a wall, add the 'passed' water symbol until anything else happens
            while self.reservoir[running] != symbols.clay:
                points.append(running)

                # Check if we fall into a hole
                below = self.reservoir[inc(running, down)]
                hole = below in (symbols.sand, symbols.passed)
                if hole:
                    # If hole, we're no longer confined, i.e. the container won't fill up further
                    confined = False
                    # If the hole is sand, we should spawn a new stream running through it
                    if below == symbols.sand:
                        new_fall_coords.append(running)
                    break

                running = inc(running, dir_)
                

        # If the area currently hit is contained (no holes), fille with water
        symbol = symbols.water if confined else symbols.passed
        for point in points:
            self.reservoir[point] = symbol
        
        if not confined:
            for new_stream_pos in new_fall_coords:
                self._spawn_child(pos=new_stream_pos)
            #
        
        # If there's no water rising, and no new child streams, this stream is done
        return bool(confined or new_fall_coords)

    def get_deepest_y(self):
        """Gets the largest y value of any child stream.
        This is to help displaying the stream which is active at the greatest depth."""

        if not self.children:
            _, y = self.pos
            res = y
        else:
            res = max((c.get_deepest_y() for c in self.children))
        
        # Update the recorded largest y value, so we don't scan back up when iteration on smaller y-vals
        if res > self._y_display_highest_returned:
            self._y_display_highest_returned = res
        else:
            res = self._y_display_highest_returned
        return res
    
    def _tick_children(self) -> bool:
        """Updates child streams, deleting any that are out of work"""
        
        for i in range(len(self.children)-1, -1, -1):
            updated = self.children[i].tick()
            if updated:
                return True
            else:
                del self.children[i]
            #
        return False

    def tick(self) -> bool:
        children_updated = self._tick_children()
        if children_updated:
            return True

        # If we're in water, back up
        try:
            self.backtrack_in_water()
        except IndexError:
            # If we run out of position history, the water has reached this stream's source, so it's done
            return False

        # Check what's below the stream. If falling down takes the stream out of bounds, it's done
        newpos = inc(self.pos, down)
        peek = self.reservoir[newpos]

        y_min, y_max = self.reservoir.y_bounds
        _, y = self.pos
        if not (y_min <= y <= y_max):
            return False

        # Determine next action based on the cell below
        match peek:
            case symbols.passed:
                # If we hit a site where water's already passed, there's nothing more to do
                return False
            case symbols.sand:
                # If we hit sand, let the stream move down one step
                self.pos = newpos
                self.reservoir[self.pos] = symbols.passed
                return True
            case symbols.water | symbols.clay:
                # If we hit water or clay, see if the water level rises, or falls off edges
                return self.fill_up()
            case _:
                raise RuntimeError  # Shouldn't happen
            #
        #
    #


def solve(data: str) -> tuple[int|str, int|str]:
    clay_sites = parse(data)
    reservoir = Reservoir(clay_sites)
    reservoir.fill(
        n_iterations=None,
        interactive_mode=False,  # For updating a single step at a time
        display_steps=False  # For printing every step
    )

    star1 = reservoir.count_water()
    print(f"Solution to part 1: {star1}")

    star2 = reservoir.count_water(valid_symbols=(symbols.water,))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 17
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
