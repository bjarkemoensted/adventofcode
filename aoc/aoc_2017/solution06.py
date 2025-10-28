# `  · .•`· .· `*.   · ·  .    ` ·. ·`  ·  .  *··   ` .+  ·   .`·.  ·  ` *·.  · 
# ·.      .·+•  .`· ·    .`·*  Memory Reallocation     ·   `·.      .·    . · `·
#  `. ·  .`    ·    .* https://adventofcode.com/2017/day/6 ·+   `·*`    .·*` ·. 
# .·  .`· +  `·  *.· `.  ·     ·.  ·  .* ·`  .·      .* ` · . ·  `.  *· ·  •.` ·


def parse(s: str):
    res = [int(elem) for elem in s.split()]
    return res


def count_redistributions(blocks):
    blocks = [val for val in blocks]
    inds = list(range(len(blocks)))
    state2first_seen = {}
    n = 0
    buffer = 0
    while True:
        signature = tuple(blocks)
        if signature in state2first_seen:
            loop_size = n - state2first_seen[signature]
            return n, loop_size

        state2first_seen[signature] = n

        ind = max(inds, key=lambda i: blocks[i])
        buffer = blocks[ind]
        blocks[ind] = 0
        while buffer:
            ind = (ind + 1) % len(blocks)
            blocks[ind] += 1
            buffer -= 1
        n += 1


def solve(data: str) -> tuple[int|str, int|str]:
    blocks = parse(data)

    n_steps, loop_size = count_redistributions(blocks)
    star1 = n_steps
    print(f"Solution to part 1: {star1}")

    star2 = loop_size
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2017, 6
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
