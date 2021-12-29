import numpy as np

with open("input22.txt") as f:
    raw = f.readlines()


def parse(line):
    """Returns parsed instructions like (1/0, [(xmin, xmax), (ymin, ...), ...])"""
    line = line.strip()
    onoff = line.split(" ")[0]
    val = {"on": 1, "off": 0}[onoff]
    lims = []
    for substring in line[len(onoff)+1:].split(","):
        intstrings = substring.split("=")[1].split("..")
        lims.append(tuple(int(s) for s in intstrings))
    res = (val, lims)
    return res


class Cube:
    def __init__(self, size=50):
        width = 2*size + 1
        self.width = width
        self.size = size
        self.m = np.zeros(shape=(width, width, width), dtype=int)

    def set(self, value, limits):
        limits_adjusted = []
        for limit in limits:
            a, b = (val + self.size for val in limit)
            if a < 0:
                a = 0
            elif a > self.width:
                return
            if b > self.width:
                b = self.width
            elif b < 0:
                return
            limits_adjusted.append((a, b+1))
        subspace_size = tuple(b-a for a, b in limits_adjusted)

        mprime = np.ones(shape=subspace_size, dtype=int)*value
        (x1, x2), (y1, y2), (z1, z2) = limits_adjusted
        self.m[x1:x2, y1:y2, z1:z2] = mprime


def determine_size(limits):
    res = float("-inf")
    for limit in sum(limits, []):
        for val in limit:
            res = max(res, abs(val))
    return res


s = """on x=-20..26,y=-36..17,z=-47..7
on x=-20..33,y=-21..23,z=-26..28
on x=-22..28,y=-29..23,z=-38..16
on x=-46..7,y=-6..46,z=-50..-1
on x=-49..1,y=-3..46,z=-24..28
on x=2..47,y=-22..22,z=-23..27
on x=-27..23,y=-28..26,z=-21..29
on x=-39..5,y=-6..47,z=-3..44
on x=-30..21,y=-8..43,z=-13..34
on x=-22..26,y=-27..20,z=-29..19
off x=-48..-32,y=26..41,z=-47..-37
on x=-12..35,y=6..50,z=-50..-2
off x=-48..-32,y=-32..-16,z=-15..-5
on x=-18..26,y=-33..15,z=-7..46
off x=-40..-22,y=-38..-28,z=23..41
on x=-16..35,y=-41..10,z=-47..6
off x=-32..-23,y=11..30,z=-14..3
on x=-49..-5,y=-3..45,z=-29..18
off x=18..30,y=-20..-8,z=-3..13
on x=-41..9,y=-7..43,z=-33..15
on x=-54112..-39298,y=-85059..-49293,z=-27449..7877
on x=967..23432,y=  373..81175,z=27513..53682""".split("\n")
instructions = [parse(line) for line in raw]

cube = Cube()

for instruction in instructions:
    cube.set(*instruction)


n_on_initialized = sum(cube.m.flat)
print(f"Solution to star 1: {n_on_initialized}.")


def find_intersection_borders(region_a, region_b):
    assert len(region_b.borders) == len(region_a.borders)
    borders = []
    for lim_a, lim_b in zip(region_a.borders, region_b.borders):
        (a1, a2), (b1, b2) = lim_a, lim_b
        p1 = max(a1, b1)
        p2 = min(a2, b2)
        if p1 > p2:
            return None
        borders.append((p1, p2))

    return borders


class Region:
    def __init__(self, borders, value=0):
        if not all(a <= b for a, b in borders):
            raise ValueError
        self.borders = borders
        self.value = value
        self.subregions = []

    def naive_area(self):
        res = 1
        for a, b in self.borders:
            res *= (1 + b - a)
        return res

    def area(self):
        res = self.naive_area() - sum(reg.naive_area() for reg in self.subregions)
        return res

    def count_on(self):
        running_regions = [self]
        sum_ = 0
        while running_regions:
            new_subregions = []
            for reg in running_regions:
                sum_ += reg.area() * reg.value
                new_subregions += reg.subregions
            running_regions = new_subregions

        return sum_

    def insert(self, other):
        running = other
        for reg in self.subregions:


    def __str__(self):
        return str(self.borders)


A = Region([(0, 10), (0, 10), (0, 10)], value=1)
B = Region([(5, 10), (5, 10), (10, 11)])

C = A.intersection(B)

A.subregions.append(C)

print(A.intersection(B))


# limits = [tup[1] for tup in instructions]
# size = determine_size(limits)
# cube2 = Cube(size=size)
# for instruction in instructions:
#     cube2.set(*instruction)
#
# n_on_initialized = sum(cube.m.flat)
# print(f"Solution to star 1: {n_on_initialized}.")