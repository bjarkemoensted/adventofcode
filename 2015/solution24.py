from itertools import chain, combinations
import numpy as np

# Read in data
with open("input24.txt") as f:
    puzzle_input = f.read()


def parse(s):
    res = []
    for line in s.split("\n"):
        res.append(int(line))

    return res


def generate_partitions(iterable, left_size):
    """Generate partitions of a specified size from input iterable.
    For example, generate_partitions([1,2,3,4,5], 2) will generate all pairs of partitions
    with 2 elements in the 'left' partition and 3 in the 'right'."""

    inds = list(range(len(iterable)))
    all_inds = set(inds)
    for comb in combinations(inds, left_size):
        left_inds = set(comb)
        right_inds = all_inds - left_inds
        left_partition = [iterable[i] for i in left_inds]
        right_partition = [iterable[i] for i in right_inds]
        yield left_partition, right_partition


def find_equipartitions(S):
    """Implements the pseudopolynomial time number partitioning algorithm.
        Stolen from here: https://en.wikipedia.org/wiki/Pseudopolynomial_time_number_partitioning"""

    n = len(S)
    K = sum(S)
    # If the sum is odd, S can't be partitioned equally
    if K % 2 != 0:
        return False

    # Initialize array to hold sub-results
    nrows = K // 2 + 1
    ncols = n + 1
    partitions = {}
    for j in range(ncols):
        # We can always find a subset which sums to zero (the empty set)
        partitions[(0, j)] = []
    for i in range(1, nrows):
        # We can never get a sum greater than zero with an empty set
        partitions[(i, 0)] = False

    for i in range(1, nrows):
        for j in range(1, ncols):
            # Apply the recurrence relation to populate the table
            x = S[j - 1]
            if i - x >= 0:
                cell = []
                # TODO I guess I should take all partitions summing to i-x and append x, and similar for the i, j-1 case
                P[i, j] = P[i, j - 1] or P[i - x, j - 1]
            else:
                P[i, j] = P[i, j - 1]
            #
        #

    return P[-1, -1]


def is_partitionable(S):
    """Implements the pseudopolynomial time number partitioning algorithm.
    Stolen from here: https://en.wikipedia.org/wiki/Pseudopolynomial_time_number_partitioning"""

    # TODO call the equipartitions method and see if any solutions exist
    n = len(S)
    K = sum(S)
    # If the sum is odd, S can't be partitioned equally
    if K % 2 != 0:
        return False

    # Initialize array to hold sub-results
    nrows = K//2 + 1
    ncols = n + 1

    P = np.empty(shape=(nrows, ncols), dtype=object)
    P[0, :] = True  # We can always find a subset which sums to zero (the empty set)
    P[1:, 0] = False  # We can never get a sum greater than zero with an empty set

    for i in range(1, nrows):
        for j in range(1, ncols):
            # Apply the recurrence relation to populate the table
            x = S[j-1]
            if i - x >= 0:
                P[i, j] = P[i, j-1] or P[i-x, j-1]
            else:
                P[i, j] = P[i, j-1]
            #
        #

    return P[-1, -1]


def determine_best_distributions(vals):
    """Determines which distributions of presents which are possible with the minimum possible number of
    packages in the passenger compartment. This may yield several results, in which case some tie-breaking
    must be applied."""

    vals = sorted(vals, reverse=True)
    solutions = []
    for n_passenger_compartment in range(1, len(vals)):
        first_combination = True
        for partitions in generate_partitions(vals, left_size=n_passenger_compartment):
            passenger, remaining = partitions
            overshoot = 2*sum(passenger) - sum(remaining)
            if first_combination and (overshoot < 0):
                break  # If the first (and largest) left configuration is too light, they'll all be.
            first_combination = False
            if overshoot != 0:
                continue  # Weight in PC must be equal to double the remaining weight, or a partition is impossible

            if is_partitionable(remaining):
                solutions.append(passenger)
            #
        if solutions:
            break
        #
    return solutions


def quantum_entanglement(arr):
    res = 1
    for x in arr:
        res *= x
    return res


weights = parse(puzzle_input)
example_weights = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]

#best_passenger_compartment_loads = determine_best_distributions(weights)
#best_qe = min(map(quantum_entanglement, best_passenger_compartment_loads))
#print(best_qe)

bla = [1, 2, 3, 4, 5, 7, 8, 10]
meh = find_equipartitions(bla)
print(meh, bla, sum(bla))