def read_input():
    with open("input03.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


class Parser:
    digits = set("0123456789")
    def type_(self, char):
        if char in self.digits:
            return "digit"
        elif char == ".":
            return "."
        else:
            return "symbol"

    def __init__(self):
        self.buffer = ""
        self.inds = []

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



def parse(s):
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


def _get_neighbors(coord):
    i, j = coord
    shift = (-1, 0, 1)
    for ishift in shift:
        for jshift in shift:
            if ishift == jshift == 0:
                continue
            yield i + ishift, j + jshift


def get_part_numbers(coord2val, coord2group):
    numbers = []
    used_groups = set([])
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


def get_gear_numbers(coord2val, coord2group):
    gear_numbers = []
    for coord, val in coord2val.items():
        if val != "*":
            continue
        neighbor_numbers = []
        used_groups = set([])
        for ncoord in _get_neighbors(coord):
            if ncoord not in coord2group or coord2group[ncoord] in used_groups:
                continue
            nval = coord2val[ncoord]
            neighbor_numbers.append(nval)
            used_groups.add(coord2group[ncoord])

        if len(neighbor_numbers) == 2:
            a, b = neighbor_numbers
            gear_numbers.append(a*b)
        #

    return gear_numbers




def main():
    raw = read_input()
    coord2val, coord2group = parse(raw)

    part_numbers = get_part_numbers(coord2val, coord2group)
    star1 = sum(part_numbers)
    print(f"Sum of engine part numbers: {star1}.")

    gear_numbers = get_gear_numbers(coord2val, coord2group)
    star2 = sum(gear_numbers)
    print(f"The gear numbers sum to: {star2}.")


if __name__ == '__main__':
    main()
