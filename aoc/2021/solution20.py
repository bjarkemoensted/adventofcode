import numpy as np

with open("input20.txt") as f:
    raw = f.read()


test_input = """..#.#..#####.#.#.#.###.##.....###.##.#..###.####..#####..#....#..#..##..##
#..######.###...####..#..#####..##..#.#####...##.#.#..#.##..#.#......#.###
.######.###.####...#.##.##..#..#..#####.....#.#....###..#.##......#.....#.
.#..#..##..#...##.######.####.####.#.#...#.......#..#.#.#...####.##.#.....
.#..#...##.#.##..#...##.#.##..###.#......#.#.......#.#.#.####.###.##...#..
...####.#..#..#.##.#....##..#.####....##...##..#...#......#.#.......#.....
..##..####..#...#.#.#...##..#.#..###..#####........#..####......#..#

#..#.
#....
##..#
..#..
..###"""

charmap = {'.': 0, '#': 1}


def parse(s):
    code, image = s.split("\n\n")
    code = "".join(code.split("\n"))
    code = [charmap[char] for char in code]
    m = np.array([[charmap[char] for char in line] for line in image.split("\n")])

    return code, m


test_code, test_m = parse(test_input)
code, m = parse(raw)


def pad(arr, n_pad=2, default=0):
    padded_shape = tuple(val + 2*n_pad for val in arr.shape)
    padded = np.ones(shape=padded_shape, dtype=int)
    padded *= default

    lastrow, lastcol = (n - n_pad for n in padded_shape)
    padded[n_pad:lastrow, n_pad:lastcol] = arr
    return padded


def lookup(subarray, code):
    assert subarray.shape == (3, 3)
    binary = sum([list(row) for row in subarray], [])
    ind = int("".join(map(str, binary)), 2)
    return code[ind]


def update(arr, code, default=0):
    n_pad = 2
    padded = pad(arr, n_pad=n_pad, default=default)
    nrows, ncols = padded.shape
    res = np.ones(shape=(nrows-n_pad, ncols-n_pad), dtype=int)
    res *= default

    for i in range(1, nrows-1):
        for j in range(1, ncols-1):
            neighborhood = padded[i-1:i+2, j-1:j+2]
            enhanced_pixel = lookup(neighborhood, code)
            res[i-1, j-1] = enhanced_pixel
        #
    return res


def update_default(default, code):
    """Takes the 'default' pixel value, i.e. the value that pixels infinitely far
    from the center have. Compute their updated values based on the conversion code."""
    default_matrix = np.ones(shape=(3, 3), dtype=int) * default
    new_default = lookup(default_matrix, code)
    return new_default


class Image:
    def __init__(self, arr, default=0):
        self.m = arr
        self.default = default

    def enhance(self, code):
        n_pad = 2
        old_default = self.default
        new_default = update_default(old_default, code)
        self.default = new_default

        padded = pad(self.m, n_pad=n_pad, default=old_default)
        nrows, ncols = padded.shape

        res = np.ones(shape=(nrows, ncols), dtype=int)
        res *= new_default

        assert res.shape == padded.shape

        for i in range(1, nrows - 1):
            for j in range(1, ncols - 1):
                neighborhood = padded[i - 1:i + 2, j - 1:j + 2]
                enhanced_pixel = lookup(neighborhood, code)
                res[i, j] = enhanced_pixel
            #
        self.m = res

    def __str__(self):
        inv = {0: "⋅", 1: "#"}
        lines = []
        for i, row in enumerate(self.m):
            line = "".join([inv[n] for n in row])
            lines.append(line)
        s = "\n".join(lines)
        return s

    def __repr__(self):
        return self.__str__()

#code = test_code
#m = test_m
img = Image(m)
img.enhance(code)
img.enhance(code)

n_light_pixels = sum(img.m.flat)

print(f"Solution to star 1: {n_light_pixels}.")

for _ in range(48):
    img.enhance(code)

n_light_pixels2 = sum(img.m.flat)
print(f"Solution to star 2: {n_light_pixels2}.")
