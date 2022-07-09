from copy import deepcopy
import numpy as np


example_input = """--- scanner 0 ---
0,2,0
4,1,0
3,3,0

--- scanner 1 ---
-1,-1,0
-5,0,0
-2,1,0
"""


def make_rotation_matrix(alpha, beta, gamma):
    cos, sin = np.cos, np.sin

    M = np.array([
        [cos(alpha)*cos(beta), cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma), cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma)],
        [sin(alpha)*cos(beta), sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma), sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma)],
        [-sin(beta), cos(beta)*sin(gamma), cos(beta)*cos(gamma)]
    ])
    return M


def make_all_roatation_matrices():
    """Generate all 24 unique 90 degree rotation matrices"""
    angles = [n*np.pi/2 for n in range(4)]
    res = []
    # Ugly hack - maintain a set of string representations of matrices already generated to avoid duplicates
    already_made = set([])
    for alpha in angles:
        for beta in angles:
            for gamma in angles:
                M = make_rotation_matrix(alpha, beta, gamma).astype(int)
                rep = str(M)
                if rep not in already_made:
                    already_made.add(rep)
                    res.append(M)

    return res


rotations = make_all_roatation_matrices()


def parse(s):
    lines = [line.strip() for line in s.split("\n") if not line.startswith("--")]
    raw = []
    buffer = []
    for line in lines:
        if line:
            thisarr = [int(elem) for elem in line.split(",")]
            buffer.append(thisarr)
        else:
            raw.append(buffer)
            buffer = []
        #
    if buffer:
        raw.append(buffer)

    res = []
    for part in raw:
        beacons = [np.array(coords) for coords in part]
        res.append(beacons)

    return res


def visualize_scan(scan, z=0):
    """Prints a visualization similar to:
    ...B.
    B....
    ....B
    S....
    """

    # List of coordinates of the beacons
    scan = deepcopy(scan)
    # Add coordinates of the scanner at the origin
    origin = np.array([0, 0, 0])
    scan.append(origin)

    # Determine the shape required to display all the coordinates
    dim = 3
    mins = [min(vec[i] for vec in scan) for i in range(dim)]
    maxes = [max(vec[i] for vec in scan) for i in range(dim)]
    shape = tuple(1 + b - a for a, b in zip(mins, maxes))

    # Make a blank 'map' array with just dots, where we'll add the beacons and scanner
    map_ = np.full(shape=shape, fill_value=".", dtype=str)

    offset = np.array(mins)
    for coords in scan:
        shifted = coords - offset
        i, j, l = shifted
        # Show 'S' for 'scanner' at origin, otherwise, 'B' for 'beacon'
        char = "S" if (coords == origin).all() else "B"
        map_[i, j, l] = char

    # Show cross section at the specified value of z
    cross_section = np.flip(map_[:, :, z].T, axis=0)
    print(cross_section)


def generate_all_offsets(scan1, scan2):
    """Return a list of offets that one could add to points in scan2 to line up points with scan1"""
    seen = set([])
    res = []
    for point1 in scan1:
        for point2 in scan2:
            offset = point1 - point2
            rep = tuple(offset)
            if rep not in seen:
                seen.add(rep)
                res.append(offset)
            #
        #
    return res


def align_scans(scan1, scan2, threshold=12):
    beacons = set([tuple(coords) for coords in scan1])
    for rotation in rotations:
        rotated = [np.dot(rotation, vec) for vec in scan2]
        for offset in generate_all_offsets(scan1, rotated):
            shifted = [vec+offset for vec in rotated]
            beacons2 = set([tuple(c) for c in shifted])
            overlap = beacons2.intersection(beacons)
            if len(overlap) >= threshold:
                return shifted, rotation, offset


def manhatten_distance(a, b):
    res = sum(abs(i - j) for i, j in zip(a, b))
    return res


scans = parse(example_input)

# Read and parse scanning data
with open("input19.txt") as f:
    data = f.read()

scans = parse(data)

# Maintain list of scans aligned with scanner 0, scans to be aligned, and the offsets applied
aligned = [scans[0]]
missing = scans[1:]
applied_offsets = []

# Repeatedly align (rotate + shift) missing scans until none remain
while missing:
    print(f"Need to align a further {len(missing)} scans.", end="\r")
    for base in aligned:
        for i in list(range(len(missing)))[::-1]:
            print(f"Need to align a further {len(missing)} scans, need to consider {i} more candidates.", end="\r")
            corrected = align_scans(base, missing[i])
            # If scans could be matched up, add the rotated+shifted coordinates to 'aligned' list and note the offset
            if corrected is not None:
                shifted, rotation, offset = corrected
                aligned.append(shifted)
                applied_offsets.append(offset)
                missing.pop(i)
            #
        #
    #
else:
    print()  # Just to leave the last printed message on terminal


# All beacon coordinates are now rotated+shifted to align with scanner 0's coordinate system. Count distinct coords.
beacons = set([])
for aligned_scan in aligned:
    for coords in aligned_scan:
        beacons.add(tuple(coords))


print(f"Number of beacons detected: {len(beacons)}.")

# List of all scanner coordinates (these are just the offsets, plus (0,0,0) for scanner 2)
scanner_locations = deepcopy(applied_offsets) + [np.array([0, 0, 0])]

# Determine the max Manhatten distance between any two scanners
max_distance = float("-inf")
for i, crd in enumerate(scanner_locations):
    for crd2 in scanner_locations[i:]:
        dist = manhatten_distance(crd, crd2)
        if dist > max_distance:
            max_distance = dist
        #
    #

print(f"Maximum Manhatten distance between any two scanners: {max_distance}.")
