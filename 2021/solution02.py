import numpy as np

with open("input02.txt") as f:
    direction_data = []
    for line in f:
        direction, val = line.strip().split()
        direction_data.append((direction, int(val)))

direction_bases = {
    'forward': np.array([1, 0]),
    'down': np.array([0, 1]),
    'up': np.array([0, -1])
}

### Star 1
position = np.array([0, 0])
for direction, distance in direction_data:
    vec = direction_bases[direction]
    position += distance * vec

prod = position[0] * position[1]
print(f"Answer to star 1: {prod}.")

### Star 2
aim = 0
position = np.array([0, 0])
for direction, x in direction_data:
    if direction == "up":
        aim -= x
    elif direction == "down":
        aim += x
    elif direction == "forward":
        position += np.array([x, x*aim])

prod2 = position[0] * position[1]
print(f"Solution to star 2: {prod2}.")
