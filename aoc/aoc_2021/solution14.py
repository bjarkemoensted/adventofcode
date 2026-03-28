# `··•.` *·  .   `··  + ·· `     *.·   +·`    · * ·  +.`· `·  *  ·`.·   * · ·`+.
# . *·`·     `.+··   `*   ·  Extended Polymerization    *·· ` ·.+ · `  .· `* +··
# ·  .· ·`    · *. `·  https://adventofcode.com/2021/day/14 .  · `   ·   * ·`·. 
# *·.`  ·    · .`   *· · +.  `· *   .· · `·    `· *.+·.·     `  ·.   *`  ·•.  ·`

from collections import Counter


def parse(s: str) -> tuple[str, dict[str, dict[str, int]]]:
    """Parses contents of input file into:
    a) starting polymer, e.g. 'NNBC',
    b) growing instructions, e.g. 'ab -> c' (which means insert c in between a's and b's),
    which is equivalent to chaning every 'ab' pair into one 'ac' and one 'cb' pair.
    Format of replacements dict is {'ab': {'ac': 1, 'cb': 1}}."""
    starting_polymer, instructions = s.split("\n\n")
    replacements = {}
    for line in instructions.split("\n"):
        # Figure out what to replacethe old string with
        old, insertion = line.strip().split(' -> ')
        pair_a = old[0] + insertion  # Make the string like 'ac'
        pair_b = insertion + old[1]  # Make the string like 'cb'
        replacement: dict[str, int] = {}
        for pair in (pair_a, pair_b):
            replacement[pair] = replacement.get(pair, 0) + 1
        replacements[old] = replacement

    return starting_polymer, replacements


def grow_polymer_sneaky(s: str, replacements: dict[str, dict[str, int]], n_iterations=10) -> dict[str, int]:
    """Grows a polymer sneakily. In each iteration, each pair (length 2 substring) grows two new pairs,
    according to the replacement data in the replacement dict.
    After n iterations, the number of occurrences of each letter is counted."""

    # Represent the polymer as all its pairs
    pairs = Counter(s[i:i+2] for i in range(len(s)-1))

    for _ in range(n_iterations):
        new_pairs: Counter[str] = Counter()
        # For each pair, grow its two descendants
        for pair, count in pairs.items():
            added_pairs = replacements[pair]
            for added_pair, weight in added_pairs.items():
                new_pairs[added_pair] += count * weight

            #
        pairs = new_pairs

    # Count each letter in the polymer
    counts: Counter[str] = Counter()
    for substring, count in pairs.items():
        for char in substring:
            counts[char] += count
        #

    # The method double counts every letter except the first and last. Compensate for this
    first, last = s[0], s[-1]
    counts[first] += 1
    counts[last] += 1
    assert all(v % 2 == 0 for v in counts.values())
    res = {k: v // 2 for k, v in counts.items()}

    return res


def compute_maxmin_diff(d: dict[str, int]) -> int:
    # Takes a dict of counts and computes the differences between the most and least common object.
    ordered = sorted(d.values())
    diff = ordered[-1] - ordered[0]
    return diff


def solve(data: str) -> tuple[int|str, ...]:
    polymer, replacements = parse(data)
    
    poly_a = grow_polymer_sneaky(polymer, replacements, 10)
    star1 = compute_maxmin_diff(poly_a)
    print(f"Solution to part 1: {star1}")

    poly_b = grow_polymer_sneaky(polymer, replacements, 40)
    star2 = compute_maxmin_diff(poly_b)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 14
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
