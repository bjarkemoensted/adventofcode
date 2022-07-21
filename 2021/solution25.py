example_input = \
"""v...>>.vv>
.vv>>.vv..
>>.>v>...v
>>v>>.>.v.
v>v.vv.v..
>.>>..v...
.vv..>.>v.
v.v..>>v.v
....v..v.>"""

with open("input25.txt", "r") as f:
    raw = f.read()


class Seabed(dict):
    """Helper class to store sea cucumber positions and no of rows+columns for use in cyclical boundary condition
    calculations."""
    def __init__(self, n_rows, n_cols, *args, **kwargs):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.shape = (n_rows, n_cols)
        super().__init__(*args, **kwargs)

    def __str__(self):
        chars = [["." for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for (i, j), char in self.items():
            chars[i][j] = char

        lines = ["".join(content) for content in chars]
        s = "\n".join(lines)
        return s


def parse(s):
    """Parses input into a seabed object"""
    lines = s.split("\n")
    n_rows = len(lines)
    n_cols = len(lines[0])
    res = Seabed(n_rows, n_cols)
    for i, line in enumerate(lines):
        for j, cucumber_type in enumerate(line):
            if cucumber_type != ".":
                pos = (i, j)
                res[pos] = cucumber_type

    return res


def get_target_square(map_shape, cucumber_type, pos):
    """Determines the location that a sea cucumber would like to move to."""
    i, j = pos
    rows, cols = map_shape
    if cucumber_type == "v":
        i = (i + 1) % rows
    elif cucumber_type == ">":
        j = (j + 1) % cols

    res = (i, j)
    return res


def tick(seabed):
    moved = False
    types = ">v"
    # Update each type of sea cucumber
    for type_ in types:
        updates = []
        for pos, char in seabed.items():
            # Skip cucumbers of the type that's not updating right now
            if char != type_:
                continue

            # If the target square is vacant, the sea cucumber will move this turn
            target = get_target_square(map_shape=seabed.shape, cucumber_type=type_, pos=pos)
            if target not in seabed:
                updates.append((pos, target, char))
                moved = True
            #
        # Update the positions for moving sea cucumbers
        for oldpos, newpos, char in updates:
            del seabed[oldpos]
            seabed[newpos] = char
    return moved


seabed = parse(raw)
done = False
n_iterations = 0
while not done:
    moved = tick(seabed)
    n_iterations += 1
    done = not moved
    print(f"Simulated {n_iterations} iterations.", end="\r")

print()

