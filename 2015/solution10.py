# Read in data
with open("input10.txt") as f:
    puzzle_input = f.read()


def extend_sequence(s):
    """Extends a sequence, e.g. transforms "11" into "21" (two ones)."""
    res = ""
    buffer = ""

    for char in s:
        if buffer and char != buffer[-1]:
            res += str(len(buffer))+buffer[-1]
            buffer = ""
        buffer += char

    if buffer:
        res += str(len(buffer)) + buffer[-1]

    return res


extended = puzzle_input
for _ in range(40):
    extended = extend_sequence(extended)


print(f"Final sequence length after 40 iterations: {len(extended)}.")

for _ in range(10):
    extended = extend_sequence(extended)

print(f"Final sequence length after 50 iterations: {len(extended)}.")
