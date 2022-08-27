import numpy as np

# Read in data
with open("input01.txt") as f:
    puzzle_input = f.read()


def parse(s):
    steps = s.strip().split(", ")
    res = []
    for step in steps:
        rotate = step[0]
        dist = int(step[1:])
        res.append((rotate, dist))

    return res


def rotate_direction(direction, rotation):
    """Rotates a direction vector left (L) or right "R" by 90 degrees."""
    rotation_matrices = {
        "R": np.array([[0, -1], [1, 0]]),
        "L": np.array([[0, 1], [-1, 0]])
    }

    M = rotation_matrices[rotation]
    res = M.dot(direction)
    return res


def trace_path(steps):
    location = np.array([0, 0])
    direction = np.array([1, 0])
    path = [tuple(location)]

    for rotation, distance in steps:
        direction = rotate_direction(direction, rotation)
        for _ in range(distance):
            location += direction
            path.append(tuple(location))

    return path


def manhatten_dist(vec):
    return sum(map(abs, vec))


instructions = parse(puzzle_input)
path = trace_path(instructions)
distance_to_final_node = manhatten_dist(path[-1])

print(f"Distance to Easter Bunny HQ: {distance_to_final_node}.")

first_doubly_visited = [path[i] for i in range(len(path)) if path[i] in path[:i]][0]
new_dist = manhatten_dist(first_doubly_visited)
print(f"Updated distance: {new_dist}.")
