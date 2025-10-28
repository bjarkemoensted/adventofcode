# ·*  . ··   `   ·  *· `··.     ·`  .  *·      .·     `·.    · ` .·* `.  *·. ·*`
#  .`··+  ` ·*  `.·· A Maze of Twisty Trampolines, All Alike `·*      ·  ··* .·+
# .··*     `+·*·    .· https://adventofcode.com/2017/day/5  .* ·`    ·`.  *··+.·
# `·+ * ·`.+ .·`·*  ` ·      *· `·.·  ` *.·   · +`· .   *· .· ·    · • `· `  · ·


import numba


def parse(s: str):
    res = [int(elem) for elem in s.splitlines()]
    return res


@numba.jit
def steps_to_complete(instructions, max_offset=-1):
    if max_offset == -1:
        max_offset = float('inf')

    instructions = [val for val in instructions]
    ind = 0
    n = 0
    while 0 <= ind < len(instructions):
        val = instructions[ind]
        shift = -1 if val >= max_offset else +1
        instructions[ind] += shift
        ind += val
        n += 1

    return n


def solve(data: str) -> tuple[int|str, int|str]:
    instructions = parse(data)

    star1 = steps_to_complete(instructions)
    print(f"Solution to part 1: {star1}")

    star2 = steps_to_complete(instructions, max_offset=3)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2017, 5
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
