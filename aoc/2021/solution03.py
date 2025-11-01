from collections import Counter

import numpy as np

# Make a matrix of ones and zeroes from input data
with open("input03.txt") as f:
    m = np.array([list(line.strip()) for line in f])

# Iterate over columns and identify most common bit
gamma_bits = []
for col in m.transpose():
    bit = max(Counter(col).items(), key=lambda t: t[1])[0]
    gamma_bits.append(bit)

gamma = "".join(gamma_bits)

# Take the negation
epsilon = "".join([{"0": "1", "1": "0"}[bit] for bit in gamma])

# Compute the product
star1 = int(gamma, 2) * int(epsilon, 2)
print(f"First solution: {star1}.")


def search_with_bit_criteria(candidates, criterion, ind=0):
    """Recursively eliminates candidate lines from the data based on whether they
    meet the input criterion. Returns the result when only one candidate remains."""
    if len(candidates) == 1:
        return "".join(candidates[0])
    # Identify target bit
    bit_counts = Counter([line[ind] for line in candidates])
    if bit_counts["0"] == bit_counts["1"]:
        # For ties, use 0 if criterion = min and 1 if criterion = max
        target_bit = criterion(bit_counts.keys())
    else:
        target_bit = criterion(bit_counts.items(), key=lambda t: t[1])[0]

    # Update candidate set
    candidates = [line for line in candidates if line[ind] == target_bit]
    # Recursive step - apply criterion to the next bit
    return search_with_bit_criteria(candidates, criterion, ind + 1)


lines = [line for line in m]
oxygen_generator_rating = search_with_bit_criteria(lines, criterion=max)

c02_scrupper_rating = search_with_bit_criteria(lines, criterion=min)

star2 = int(oxygen_generator_rating, 2) * int(c02_scrupper_rating, 2)
print(f"Second solution: {star2}.")
