# ·.  ·*·`•   · .` ·· . ·*`+  ·· `*·. ·     ·• *` ·.* ·`    + ·*·` . ` *·  ·`.·*
# `*·· .*  `.  ·*  ` ·*· ·.   ` * Bridge Repair .* `·   *    · ·`  ·*. · ` . *`·
# · `. ·  ·*   + * ·   https://adventofcode.com/2024/day/7 .·` ·  *  * . · `·· .
# .  +·` . ·•·  ··.* ` *   ··    ·  * .` · . *  · *  `.·   · .   ·` · · * ·+` .`


def parse(s: str):
    """Parses into tuples like (result, [component1, component2, ...])"""

    res = []
    for line in s.splitlines():
        a_s, b_s = line.split(":")
        a = int(a_s)
        b = [int(part) for part in b_s.strip().split()]
        res.append((a, b))

    return res


def is_solvable(result: int, terms: list, operations, running: int=0):
    """Determines recursively whether the result can be obtained by applying the input operations to
    the input terms in any order. Assumes that the running result only increases by applying operations."""

    # Stop if we overshoot
    if running > result:
        return False
    
    # If we run out of terms, the running result must equal the final result
    if not terms:
        return running == result
    
    elem = terms[0]
    rest = terms[1:]
    
    # On the first iteration, set the running result to the first term
    if running == 0:
        return is_solvable(result, rest, operations, running=elem)
    
    # recursion step - try all operations with updated running result on the remaining terms
    subsequent = (is_solvable(result, rest, operations, op(running, elem)) for op in operations)
    
    return any(subsequent)


def solve(data: str) -> tuple[int|str, int|str]:
    equations = parse(data)
    
    operations = [
        lambda a, b: a+b,
        lambda a, b: a*b
    ]
    
    star1 = sum(res for res, terms in equations if is_solvable(res, terms, operations))
    print(f"Solution to part 1: {star1}")

    operations.append(lambda a, b: int(f"{a}{b}"))
    star2 = sum(res for res, terms in equations if is_solvable(res, terms, operations))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 7
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
