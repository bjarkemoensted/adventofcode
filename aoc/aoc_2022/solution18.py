# *`.  ·`·  +  ·`   •·   .· `  *··  ·+. ` `·   ·    +`  ·.   · `*·   . •`· ` ·.`
# ··  *`..  · `+·.     ·.·*` · ` Boiling Boulders ` .· +.·  `  ·   .*  ·· . · `·
# . ·`·  *·   · +`.    https://adventofcode.com/2022/day/18   `    ·  . .*`··+·.
#  .`·.+ ` · `* .· *`· +  .·   ·   ·.`   `.*·`  · ·.• `·   . *·  •    ·`•.·*`` ·

from typing import TypeGuard

type voxeltype = tuple[int, int, int]


def is_voxel(obj) -> TypeGuard[voxeltype]:
    return isinstance(obj, tuple) and len(obj) == 3 and all(isinstance(elem, int) for elem in obj)


def parse(s: str) -> list[voxeltype]:
    res = []
    for line in s.splitlines():
        x, y, z = map(int, line.split(","))
        res.append((x, y, z))

    return res


def neighbor_sites(voxel: voxeltype) -> list[voxeltype]:
    """Takes a voxel (x,y,z tuple) and returns all neighbor sites"""
    res = []
    for ind in range(len(voxel)):
        for delta in (-1, 1):
            arr = [val for val in voxel]
            arr[ind] += delta
            neighbor = tuple(arr)
            assert is_voxel(neighbor)
            res.append(neighbor)
        #

    return res


def compute_surface_area(*voxels: voxeltype) -> int:
    """Computes the surface area as all voxel surfaces that do not touch another voxel."""
    all_voxels = set(voxels)
    area = 0
    for voxel in voxels:
        for neighbor in neighbor_sites(voxel):
            area += neighbor not in all_voxels
        #

    return area


def determine_shell(*voxels: voxeltype) -> set[voxeltype]:
    """Determines the set of voxels that makes up a 'shell' around the input voxels, i.e. the voxels spanning the input
    range of x, y, and z-values but not any volume sealed off inside the voxels. Found by BFS from an outer corner."""

    voxels_set = set(voxels)

    # Determine the bounds
    xyz = list(zip(*voxels_set))
    low = [min(arr) - 1 for arr in xyz]
    high = [max(arr) + 1 for arr in xyz]

    # Make a voxel set for the results, and one for the 'front' of voxels being added in the BFS procedure
    starting_point = tuple(low)
    res = {starting_point}
    new_air_sites = {starting_point}
    while new_air_sites:
        add_ = set([])  # New voxels to add in this iteration
        for new_air in new_air_sites:
            for neighbor in neighbor_sites(new_air):
                # We can use the new voxel if it's within bounds, not in the input voxels, and not already seen
                within_bounds = all(a <= val <= b for a, val, b in zip(low, neighbor, high))
                within_lava = neighbor in voxels_set
                already_seen = neighbor in res
                valid = within_bounds and not(within_lava or already_seen)
                if valid:
                    add_.add(neighbor)
                #
            #
        res.update(new_air_sites)
        new_air_sites = add_

    return res


def determine_outer_surface_area(*voxels: voxeltype):
    """Determines the outer surface area of a voxel set. Finds the surface shared by input voxels and the outer 'shell'
    sorrounding them."""

    air = determine_shell(*voxels)
    area = 0
    for voxel in voxels:
        for neighbor in neighbor_sites(voxel):
            area += neighbor in air
        #

    return area


def solve(data: str) -> tuple[int|str, ...]:
    voxels = parse(data)

    star1 = compute_surface_area(*voxels)
    print(f"Solution to part 1: {star1}")

    star2 = determine_outer_surface_area(*voxels)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 18
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
