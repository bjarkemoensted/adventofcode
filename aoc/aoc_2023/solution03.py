# `··. •· `·   .+··` .+     ··` · .  ·• *   ·     ··**   .`··•    ·    .·``·*·. 
# ·  *·`  ·.•· * `.·· ·`.·  *    · Gear Ratios `·  .`·*·   •`      *.·` `·+*. ··
# ··`   ·*.`    · +` * https://adventofcode.com/2023/day/3      ·  ·`. `  ·`·. +
# *` ·. `·* ·`·  .  *·  ·*.· `     ·    ·`·  *.     · . ·`  ·.· •·*` ·    . `+··


from typing import Iterator, Literal, TypeAlias

cattype: TypeAlias = Literal["digit", ".", "symbol"]
coordtype: TypeAlias = tuple[int, int]
valmap: TypeAlias = dict[coordtype, int]
groupmap: TypeAlias = dict[coordtype, tuple[coordtype, ...]]


class Parser:
    digits = set("0123456789")
    
    def type_(self, char: str) -> cattype:
        if char in self.digits:
            return "digit"
        elif char == ".":
            return "."
        else:
            return "symbol"

    def __init__(self) -> None:
        self.buffer = ""
        self.inds: list[tuple[int, int]] = []

    def flush(self):
        if not self.buffer:
            return None
        try:
            val = int(self.buffer)
        except ValueError:
            val = self.buffer

        inds = [tup for tup in self.inds]
        self.buffer = ""
        self.inds = []
        return inds, val

    def readchar(self, c, ind):
        res = None
        buffer_needs_flushing = self.buffer and self.type_(self.buffer[-1]) != self.type_(c)
        if buffer_needs_flushing:
            res = self.flush()

        if self.type_(c) != ".":
            self.buffer += c
            self.inds.append(ind)
        return res


def parse(s) -> tuple[valmap, groupmap]:
    coord2val = {}
    coord2group = {}
    parser = Parser()

    def register(val):
        nonlocal  coord2val, coord2group
        if val is None:
            return
        coords, x = val
        all_coords = tuple(coords)
        for coord in coords:
            coord2val[coord] = x
            coord2group[coord] = all_coords

    for i, line in enumerate(s.split("\n")):
        for j, char in enumerate(line):
            ind = (i, j)
            val = parser.readchar(char, ind)
            register(val)

        val = parser.flush()
        register(val)
    
    return coord2val, coord2group


def _get_neighbors(coord: coordtype) -> Iterator[coordtype]:
    """Provides the 8 sites surrounding the input coordinate"""
    i, j = coord
    shift = (-1, 0, 1)
    for ishift in shift:
        for jshift in shift:
            if ishift == jshift == 0:
                continue
            yield i + ishift, j + jshift
        #
    #


def get_part_numbers(coord2val: valmap, coord2group: groupmap) -> list[int]:
    """Scans across all coordinates, and looks for part numbers in their vicinity"""
    
    numbers: list[int] = []
    used_groups: set[tuple[coordtype, ...]] = set([])
    for coord, val in coord2val.items():
        if isinstance(val, int):
            continue

        for ncoord in _get_neighbors(coord):
            try:
                nval = coord2val[ncoord]
                ngroup = coord2group[ncoord]
            except KeyError:
                continue
            
            if isinstance(nval, int) and ngroup not in used_groups:
                numbers.append(nval)
                used_groups.add(ngroup)
            #
        #
    return numbers


def get_gear_numbers(coord2val: valmap, coord2group: groupmap) -> list[int]:
    """Returns a list of all gear numbers, found by scanning across all coordinates,
    and extracting values around each gear symbol ('*')"""

    gear_numbers: list[int] = []
    for coord, val in coord2val.items():
        # Only consider the neighborhood around gears
        if val != "*":
            continue
        
        # Grab the values around the current gear
        neighbor_numbers = []
        used_groups = set([])
        for ncoord in _get_neighbors(coord):
            if ncoord not in coord2group or coord2group[ncoord] in used_groups:
                continue
            nval = coord2val[ncoord]
            neighbor_numbers.append(nval)
            used_groups.add(coord2group[ncoord])
        
        # If the gear connects exactly 2 values, keep their product
        if len(neighbor_numbers) == 2:
            a, b = neighbor_numbers
            gear_numbers.append(a*b)
        #

    return gear_numbers


def solve(data: str) -> tuple[int|str, ...]:
    coord2val, coord2group = parse(data)

    part_numbers = get_part_numbers(coord2val, coord2group)
    star1 = sum(part_numbers)
    print(f"Solution to part 1: {star1}")

    gear_numbers = get_gear_numbers(coord2val, coord2group)
    star2 = sum(gear_numbers)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 3
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
