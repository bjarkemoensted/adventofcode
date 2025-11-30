# ·..  `* ··   .* `.  ··.*` `·+  .  ·  +.*`· `+.·.`• ·        .·`*   ·. ··* . *·
# .·`··* . +·. · •· . * ·`. · . · Encoding Error · `  .·* ·.  ·*  `+ .·* ` · ··.
# *· .·· `•. ·  ·    · https://adventofcode.com/2020/day/9 ·  ` *·. ·  ·. ·+`*. 
# ` ·  .··.   .*`· ·+`.  ·`  • ·.  · .  ··*  · `· .  *·.`·+  ·*.   ·.  ·`.  ·`+·


def parse(s: str) -> list[int]:
    res = [int(line.strip()) for line in s.splitlines()]
    return res


def find_first_invalid(numbers: list[int], preamble_len = 25) -> int:
    """Finds the first invalid number in the list"""
    for idx in range(preamble_len, len(numbers)):
        preamble = set(numbers[idx - preamble_len: idx])
        num = numbers[idx]

        valid = False
        for a in preamble:
            b = num - a
            if b in preamble:
                valid = True
            #
        if not valid:
            return num
        #
    
    raise RuntimeError


def find_contiguous_sum(numbers: list[int], target: int) -> list[int]:
    """Find a range of subsequent elements which sum to the target value"""
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            contiguous = numbers[i:j]
            sum_ = sum(contiguous)
            if sum_ == target:
                return contiguous
            if sum_ > target:
                break
            #
        #
    raise RuntimeError


def solve(data: str) -> tuple[int|str, ...]:
    numbers = parse(data)

    star1 = find_first_invalid(numbers, preamble_len=25)
    print(f"Solution to part 1: {star1}")

    contiguous_nums = find_contiguous_sum(numbers, target=star1)
    star2 = min(contiguous_nums) + max(contiguous_nums)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 9
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
