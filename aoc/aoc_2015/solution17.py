# .·•  `·  .  ` ·.  *    * ·   · .  ··`   *·.+    ··  . ·`  * · .*  ·.` +·* · ·.
# •. ·`·*  ·. . * ·`·  *` · No Such Thing as Too Much ·` *·.  `  ·`·.  + ` *.  ·
#  ·. * `*  · · .*   ` https://adventofcode.com/2015/day/17 ·`+   .`·   ·*`. · `
# ·*`• .· ·`   ·  •·`· . ·*`. ·`* +·. `  *·  ·`.   *· + `· * . `*.· *  ·`   •`·.


from collections import Counter
import math


def parse(s: str):
    res = [int(line) for line in s.split("\n")]
    return sorted(res)


def extend(used_containers, remaining_containers, running_sum, target_volume):
    """Extends a list of possible combinations of containers which sum to target volume.
    user_containers: A list of the containers used in this combination so far.
    remaining_containers: Containers that may be added to the list.
    running_sum: The volume of containers used so far.
    target_volume: The volume which the used containers must sum to."""

    res = []
    # Try extending the used list with every possible new addition
    for i, new_container in enumerate(remaining_containers):
        # Enforce that containers must be ordered by size, so we don't double count
        if used_containers and new_container < used_containers[-1]:
            continue

        # Add the new container and compute the resulting volume
        new_sum = running_sum + new_container
        new_used = [c for c in used_containers] + [new_container]

        # If we exceed the target volume, adding the new container is no good
        if new_sum > target_volume:
            continue

        # If we reach the target, we can use the newly added container
        if new_sum == target_volume:
            res.append(new_used)
        # If we're not there yet, recursively try adding new ones
        else:
            new_remaining = [remaining_containers[ind] for ind in range(len(remaining_containers)) if ind != i]
            res += extend(new_used, new_remaining, new_sum, target_volume)
        #
    return res


def brute_force(containers, target_volume):
    """Brute force all (ordered) combinations of containers summing to the specified volume"""
    res = extend(
        used_containers=[],
        remaining_containers=containers,
        running_sum=0,
        target_volume=target_volume
    )

    return res


def count_unique_combinations(combinations):
    """Counts unique combinations, e.g. (5, 5, 15) will occur twice (two permutations of the '5' containers),
    so is weighted with 1/2."""
    res = 0.0
    for comb in combinations:
        # Inversely weight with the number of possible permutations
        n_permutations = 1
        for k, v in Counter(comb).items():
            n_permutations *= math.factorial(v)
        res += 1.0/n_permutations

    return round(res)


def solve(data: str) -> tuple[int|str, int|str]:
    containers_puzzle = parse(data)
    target_volume = 150
    # Find the number of combinations of containers that have a combined volume of 150L.
    combinations = brute_force(containers_puzzle, target_volume=target_volume)
    star1 = count_unique_combinations(combinations)
    print(f"Solution to part 1: {star1}")

    # Find the number of such combinations which uses the lowest possible number of containers
    lowest_n_containers = min(map(len, combinations))
    short_combs = [comb for comb in combinations if len(comb) == lowest_n_containers]
    star2 = count_unique_combinations(short_combs)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 17
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
