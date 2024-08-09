from itertools import combinations
import numpy as np


def parse(s):
    res = []
    for line in s.split("\n"):
        res.append(int(line))

    return res


def complement(numbers, subset):
    """Returns the complement to a subset given numbers.
    Example: numbers=[1,2,3], subset=[2] -> [1, 3]"""

    if not all(isinstance(val, int) for val in numbers):
        raise TypeError
    if not all(isinstance(val, int) for val in subset):
        raise TypeError

    subset = set(subset)
    res = [number for number in numbers if number not in subset]
    return res


def iterate_subsets(numbers, required_sum):
    """Iterates over all subsets of input numbers, which has the specified sum."""

    # Ensure numbers are sorted descending
    numbers = sorted(numbers, reverse=True)
    for n_elems in range(1, len(numbers)):
        for subset in combinations(numbers, n_elems):
            sum_ = sum(subset)
            if sum_ == required_sum:
                yield subset


def subset_sum(numbers, target):
    """Implements the pseudopolynomial time number partitioning algorithm for the subset sum problem.
    Stolen from here: https://en.wikipedia.org/wiki/Pseudopolynomial_time_number_partitioning"""

    n = len(numbers)

    # Initialize array to hold sub-results
    nrows = target + 1
    ncols = n + 1

    P = np.empty(shape=(nrows, ncols), dtype=object)
    P[0, :] = True  # We can always find a subset which sums to zero (the empty set)
    P[1:, 0] = False  # We can never get a sum greater than zero with an empty set

    for i in range(1, nrows):
        for j in range(1, ncols):
            # Apply the recurrence relation to populate the table
            x = numbers[j-1]
            if i - x >= 0:
                P[i, j] = P[i, j-1] or P[i-x, j-1]
            else:
                P[i, j] = P[i, j-1]
            #
        #

    return P[-1, -1]


def is_partitionable(numbers, n_parts):
    """Determines if input numbers can be partitioned into n parts with equal sums. Example:
    numbers=[1,2,3,6], n_parts=3 -> True"""

    sum_ = sum(numbers)
    if sum_ % n_parts != 0:
        raise ValueError
    target_sum = sum_ // n_parts

    # Any set of numbers sums to its own sum (duh)
    if n_parts == 1:
        return True

    # For partitioning into two parts, use dynamic programming tricks
    if n_parts == 2:
        return subset_sum(numbers, target_sum)

    # Otherwise, brute force for subsets summing to sum/n, then recurse on the remaining numbers
    for subset in iterate_subsets(numbers, required_sum=target_sum):
        remaining = complement(numbers, subset)
        # If recursion suceeds, we've found a partition that works
        if is_partitionable(remaining, n_parts-1):
            return True
        #

    # If we never found a solution, no partition exists
    return False


def quantum_entanglement(arr):
    """Computes the 'quantum entanglement' (product of numbers)."""
    res = 1
    for x in arr:
        res *= x
    return res


def balance_loads(numbers, n_partitions):
    """Determines the optimal load for the input weights. 'Optimal' here means minimum number of weights in the first
    partition, using 'quantum entanglement' as tiebreaker.
    Returns the QE of the optimal configuration."""

    solutions = []
    target_sum = sum(numbers) // n_partitions
    current_solution_length = -1

    for subset in iterate_subsets(numbers, required_sum=target_sum):
        # We need the shortes possible subset, so if we starting iterating over a new length, stop if solutions exist
        new_length = len(subset)
        if new_length != current_solution_length:
            current_solution_length = new_length
            if solutions:
                break

        # if the remaining numbers can be partitioned into n-1 partitions, this subset is a solution
        remaining_numbers = complement(numbers, subset)
        if is_partitionable(remaining_numbers, n_partitions-1):
            solutions.append(subset)

    # The solution with the minimum QE is the best solution
    quantum_entanglements = map(quantum_entanglement, solutions)
    res = min(quantum_entanglements)

    return res


def solve(data: str):
    weights = parse(data)

    star1 = balance_loads(weights, n_partitions=3)
    print(f"Solution to part 1: {star1}")

    star2 = balance_loads(weights, n_partitions=4)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2015, 24
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
