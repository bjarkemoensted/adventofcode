#  ꞏ*` * `.+⸳ `•ꞏ   ` *.  •⸳ꞏ`⸳ .* * ⸳ `ꞏ  •*.     . `ꞏ    *`ꞏ+ . ⸳•   `  *`• ⸳ꞏ
#  `ꞏ* ꞏ . ⸳ ꞏ+•         `ꞏ+ *` Red-Nosed Reports .*+       ⸳` +`ꞏ*⸳ꞏ   `    •ꞏ*
#  *  . `⸳*. +ꞏ  *⸳    https://adventofcode.com/2024/day/2  +* ꞏ ⸳     *ꞏ ` *  ⸳
# ꞏ ⸳•` ꞏ* *`  ꞏ  `. *ꞏ `•    ꞏ*.⸳ ⸳ **ꞏ*`ꞏ    ꞏ.  ` * *ꞏ •  .`   *ꞏ⸳. `*⸳ * ꞏ`*


def parse(s):
    res = []
    for line in s.splitlines():
        report = [int(elem) for elem in line.split()]
        res.append(report)
    return res


def report_is_safe(report: list) -> bool:
    """Checks if a report (list of numbers from the reindeer reactor thingy) is safe"""
    
    # Set the max difference between two subsequent numbers
    low = 1
    high = 3
    # Flip if list is decreasing
    increasing = report[1] > report[0]
    if not increasing:
        low, high = -high, -low
    
    for i in range(len(report) - 1):
        diff = report[i + 1] - report[i]
        if not (low <= diff <= high):
            return False
        #
    
    return True


def brute_check(report):
    """Checks if a report can be safe be removing at most one element from the list"""
    
    # Nothing to do if it's already safe
    if report_is_safe(report):
        return True
    
    # Otherwise, just try removing each value and see if that works
    for i in range(len(report)):
        if report_is_safe(report[:i]+report[i+1:]):
            return True
        #
    
    return False


def solve(data: str):
    reports = parse(data)

    star1 = sum(report_is_safe(report) for report in reports)
    print(f"Solution to part 1: {star1}")

    star2 = sum(brute_check(report) for report in reports)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2024, 2
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
