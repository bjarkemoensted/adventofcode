def read_input():
    with open("input18.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = [tuple(int(substring) for substring in line.split(",")) for line in s.split("\n")]
    return res


def neighbor_sites(voxel):
    """Takes a voxel (x,y,z tuple) and returns all neighbor sites"""
    res = []
    for ind in range(len(voxel)):
        for delta in (-1, 1):
            arr = [val for val in voxel]
            arr[ind] += delta
            res.append(tuple(arr))
        #

    return res


def compute_surface_area(voxels):
    """Computes the surface area as all voxel surfaces that do not touch another voxel."""
    all_voxels = set(voxels)
    area = 0
    for voxel in voxels:
        for neighbor in neighbor_sites(voxel):
            area += neighbor not in all_voxels
        #

    return area


def determine_shell(voxels):
    """Determines the set of voxels that makes up a 'shell' around the input voxels, i.e. the voxels spanning the input
    range of x, y, and z-values but not any volume sealed off inside the voxels. Found by BFS from an outer corner."""

    voxels = set(voxels)

    # Determine the bounds
    xyz = list(zip(*voxels))
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
                within_lava = neighbor in voxels
                already_seen = neighbor in res
                valid = within_bounds and not(within_lava or already_seen)
                if valid:
                    add_.add(neighbor)
                #
            #
        res.update(new_air_sites)
        new_air_sites = add_

    return res


def determine_outer_surface_area(voxels):
    """Determines the outer surface area of a voxel set. Finds the surface shared by input voxels and the outer 'shell'
    sorrounding them."""

    air = determine_shell(voxels)
    area = 0
    for voxel in voxels:
        for neighbor in neighbor_sites(voxel):
            area += neighbor in air
        #

    return area


def main():
    raw = read_input()
    voxels = parse(raw)

    area = compute_surface_area(voxels)
    print(f"Lava droplet has surface area {area}.")

    outer_area = determine_outer_surface_area(voxels)
    print(f"Outer surface area of lava droplet is {outer_area}.")


if __name__ == '__main__':
    main()
