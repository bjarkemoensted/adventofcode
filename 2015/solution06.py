import numpy as np

# Read in data
with open("input06.txt") as f:
    raw = f.read()


def parse(stuff):
    """Parses input into a list of intructions in the form
    (instruction, coords1, coords2)."""

    res = []

    for line in stuff.split("\n"):
        parts = line.split(" ")
        coords = [parts[-3], parts[-1]]

        coords1, coords2 = [tuple(int(elem) for elem in s.split(",")) for s in coords]
        instruction = " ".join(parts[:-3])

        res.append((instruction, coords1, coords2))
    return res


def update(arr, operation):
    """Updates the array of Christmas lights when lights are turned on/off or toggled between on/off.
    Toggling is implemented as incrementing by 1 - we can just take the value mod 2 later on."""

    instruction, coords1, coords2 = operation
    (i1, j1), (i2, j2) = coords1, coords2

    if instruction == "turn on":
        arr[i1:i2+1, j1:j2+1] = 1
    elif instruction == "turn off":
        arr[i1:i2+1, j1:j2+1] = 0
    elif instruction == "toggle":
        arr[i1:i2 + 1, j1:j2 + 1] += 1
    else:
        raise ValueError('Invalid instruction')


lights = np.zeros(shape=(1000, 1000), dtype=int)
instructions = parse(raw)

for ins in instructions:
    update(lights, ins)

print(f"There are {sum(v % 2 for v in lights.flat)} lights on.")


def update2(arr, operation):
    """Updates the array of Christmas lights using the new rules for brightness."""

    instruction, coords1, coords2 = operation
    (i1, j1), (i2, j2) = coords1, coords2

    if instruction == "turn on":
        arr[i1:i2+1, j1:j2+1] += 1
    elif instruction == "turn off":
        arr[i1:i2+1, j1:j2+1] = arr[i1:i2+1, j1:j2+1] - 1
        # Set negative brightnesses to zero
        arr[i1:i2+1, j1:j2+1] *= arr[i1:i2 + 1, j1:j2 + 1] >= 0
    elif instruction == "toggle":
        arr[i1:i2 + 1, j1:j2 + 1] += 2
    else:
        raise ValueError('Invalid instruction')


lights = np.zeros(shape=(1000, 1000), dtype=int)

for ins in instructions:
    update2(lights, ins)

print(f"Total brightness is {sum(lights.flat)}.")
