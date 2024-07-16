from copy import deepcopy

with open("input22.txt") as f:
    raw = f.read()


def parse(s):
    """Returns parsed instructions like (1/0, [(xmin, xmax), (ymin, ...), ...])"""
    res = []
    for line in s.split("\n"):
        line = line.strip()
        onoff = line.split(" ")[0]
        val = {"on": 1, "off": 0}[onoff]
        lims = []
        for substring in line[len(onoff) + 1:].split(","):
            intstrings = substring.split("=")[1].split("..")
            lims.append(tuple(int(s) for s in intstrings))
        res.append((val, lims))
    return res


class Cube:
    """Cube object. Holds the limits in the x, y, and z directions, and has helper methods for string representation
    and volume computations."""
    def __init__(self, limits):
        self.limits = limits

    def volume(self):
        res = 1
        for a, b in self.limits:
            res *= (b + 1 - a)
        return res

    def __str__(self):
        res = f"Cube at ({', '.join(map(str, self.limits))}, V={self.volume()})"
        return res

    def __repr__(self):
        return self.__str__()


def get_overlap(x, y):
    """Returns the overlap between cubes x and y. Returns None if they don't overlap"""
    limits = []
    for (a1, a2), (b1, b2) in zip(x.limits, y.limits):
        low = max(a1, b1)
        high = min(a2, b2)
        if low > high:
            return None
        limits.append((low, high))
    res = Cube(limits)
    return res


def dice(cube, cutout):
    """Takes two cube-shaped objects. Returns a list of cubes that together make up the
    volume of input cube with the cutout cube removed."""
    assert len(cutout.limits) == len(cube.limits)

    # Remaining part of the input cube (shrinks as more and more volume is sliced away)
    remaining = deepcopy(cube)
    # If there's no overlap, the result is simply the initial cube
    overlap = get_overlap(cube, cutout)
    if not overlap:
        return [remaining]

    # Compute the volume of the result for sanity checking
    initial_volume = cube.volume()
    target_volume = initial_volume - overlap.volume()

    parts = []
    for i in range(len(cutout.limits)):
        a_cut, b_cut = cutout.limits[i]
        a, b = remaining.limits[i]
        if a < a_cut:
            # See if we can slice of a section below the cutout
            cut_here = [(a, a_cut - 1) if ii == i else lim for ii, lim in enumerate(remaining.limits)]
            parts.append(Cube(cut_here))
            a = a_cut
            remaining.limits[i] = (a, b)
        if b > b_cut:
            # Aaand cut above
            cut_here = [(b_cut + 1, b) if ii == i else lim for ii, lim in enumerate(remaining.limits)]
            parts.append(Cube(cut_here))
            b = b_cut
            remaining.limits[i] = (a, b)
        #

    # Check that we got the volume right
    final_volume = sum(part.volume() for part in parts)
    if not final_volume == target_volume:
        raise ValueError

    return parts


def shatter(hard_cubes, soft_cubes):
    """Takes a list of 'hard' and 'soft' cubes. Iteratively cuts from every soft cube the overlap between it and
    the hard cubes. Returns a list cubes spanning the remaining volume."""
    running = deepcopy(soft_cubes)
    for hard in hard_cubes:
        new_soft_cubes = []  # Holds the soft cubes for subsequent iteration
        for soft in running:
            parts = dice(soft, hard)
            new_soft_cubes += parts
        running = new_soft_cubes
    return running


def run_instructions(steps):
    """Runs the input instructions"""
    cubes = []  # Holds all cubes that are turned on
    for value, limits in steps:
        new_cube = Cube(limits)
        # Identify existing cubes which overlaps with the new one
        overlap_inds = [i for i, cube in enumerate(cubes) if get_overlap(cube, new_cube) is not None]
        if value == 1:
            # If we're turning on a region, ignore the regions that are already on
            soft = [new_cube]
            hard = [cubes[i] for i in overlap_inds]
            # Remove from new cube the parts that we already have
            shattered = shatter(hard, soft)
            # Add resulting parts
            cubes += shattered
        elif value == 0:
            # If we're turning off a region, remove overlaps with existing cubes
            # Take out the affected regions
            soft = []
            for i in overlap_inds[::-1]:
                soft.append(cubes.pop(i))
            # Remove the parts that we have to turn off
            hard = [new_cube]
            shattered = shatter(hard, soft)
            # Put the remainder back in
            cubes += shattered
        #
    return cubes


# Read in instructions
instructions = parse(raw)
# Identify boot instructions (cubes that are confined to x,y,z in [-50:50])
boot_instructions = [(val, limits) for val, limits in instructions if all(a >= -50 and b <= 50 for a, b in limits)]

# Find volume of region turned on after booting up
boot_cubes = run_instructions(boot_instructions)
boot_volume = sum(cube.volume() for cube in boot_cubes)
print(f"Solution to star 1: {boot_volume}.")

# Find volume of region turned on after running the full instructions
final_cubes = run_instructions(instructions)
final_volume = sum(cube.volume() for cube in final_cubes)
print(f"Solution to star 2: {final_volume}.")
