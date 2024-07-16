def read_input():
    with open("input22.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = []
    for line in s.split("\n"):
        cube = tuple(tuple(map(int, elem.split(","))) for elem in line.split("~"))
        res.append(cube)

    assert all(all(a <= b for a, b in zip(*cube)) for cube in res)

    return res


_ground = ((float("-inf"), float("-inf"), 0), (float("inf"), float("inf"), 0))
_down = (0, 0, -1)


def collision(cube1, cube2):
    """Returns true if the two cubes collide with each other."""
    dim = len(cube1[0])

    # Assume collision. Switch to False if any coordinate (x, y, z) is disjunct between the two cubes
    res = True
    for i in range(dim):
        low1 = cube1[0][i]
        high1 = cube1[1][i]
        low2 = cube2[0][i]
        high2 = cube2[1][i]
        disjunct = low2 > high1 or high2 < low1
        if disjunct:
            return False
        #
    return res


def _shift_coords(cube, dir_):
    res = tuple(tuple(a + b for a, b in zip(corner, dir_)) for corner in cube)
    return res


def trymove(cube, obstacles, dir_=None, repeat=False):
    """Tries moving a cube in the specified direction. Returns new location if successful. Returns old location
    if the new position is blocked by an obstacle or by the ground.
    if repeat, keep repeating until a collision is detected."""

    if dir_ is None:
        dir_ = _down

    newcube = _shift_coords(cube, dir_)

    if collision(newcube, _ground) or any(collision(newcube, obstacle) for obstacle in obstacles):
        return cube
    else:
        if repeat:
            return trymove(newcube, obstacles, dir_=dir_, repeat=repeat)
        else:
            return newcube


def fall(cubes):
    """Let all the cubes fall into place. Starting with the cube at the lowest z-value."""

    cubes_done = []
    for i, cube in enumerate(sorted(cubes, key=lambda cube: cube[0][2])):
        cube_landed = trymove(cube, obstacles=cubes_done, repeat=True)
        cubes_done.append(cube_landed)
        print(f"{i+1} of {len(cubes)} cubes have fallen into place.", end="\r")
    print()

    return cubes_done


def make_restmap(cubes):
    """Maps all cubes (index) to the indices of the cubes on which they rest.
    For example, {1: [2,3,4], ...} indicates that cube 1 rests on cubes 2, 3, and 4.
    Works by trying to shift all cubes downwards one step, then identifying with which cubes it collides."""

    res = {}
    for i, cube in enumerate(cubes):
        res[i] = []
        newcube = _shift_coords(cube, dir_=_down)
        for j, othercube in enumerate(cubes):
            if i == j:
                continue
            if collision(newcube, othercube):
                res[i] += [j]
            #
        #
    return res


def find_disintegrable_cubes(restmap):
    """Identifies the cubes which may be safely disintegrated."""

    # Start with the set of all cubes
    all_cubes = set(restmap.keys())

    # Disqualify any cubes that are the sole support for another cube
    sole_supports = set([])
    for supports in restmap.values():
        if len(supports) == 1:
            sole_supports.add(supports[0])
        #

    # Cubes that are not sole supporters are safe to disintegrate
    res = all_cubes - sole_supports
    return res


def count_chain_reaction_cubes(restmap):
    """Counts the number of cubes which can be disintegrated as a result of a chain reaction.
    Works by trying to start a chain reaction from each cube and observing the number of cubes destroyed in the
    subsequent chain reaction."""

    res = 0
    for cube in restmap.keys():
        # Repeatedly destroy cubes which just lost all their supports
        destroyed = set([])
        boom = {cube}
        while boom:
            destroyed = destroyed.union(boom)
            boom = set([])
            for newcube, supports in restmap.items():
                if supports and all(support in destroyed for support in supports) and newcube not in destroyed:
                    boom.add(newcube)
                #
            #
        # Only count the cubes destroyed from the chain reaction, not the initial destruction
        res += len(destroyed) - 1

    return res


def main():
    raw = read_input()
    cubes = parse(raw)

    cubes_on_ground = fall(cubes)
    restmap = make_restmap(cubes_on_ground)
    discubes = find_disintegrable_cubes(restmap)
    star1 = len(discubes)
    print(f"{star1} cubes can be safely disintegrated.")

    star2 = count_chain_reaction_cubes(restmap)
    print(f"Chain reactions can disintegrate {star2} bricks.")


if __name__ == '__main__':
    main()
