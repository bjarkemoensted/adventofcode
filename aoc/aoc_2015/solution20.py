import math
import numba


def parse(s):
    res = int(s)
    return res


@numba.njit(cache=True)
def find_divisors(n: int):
    if n == 1:
        return [1]

    # Look for ints thru sqrt(n) which divides n
    low = 2
    high = math.floor(math.sqrt(n) + 1)
    for i in range(low, high):
        # If we find one, use the cached divisors from n/i
        if n % i == 0:
            n_div = n // i
            old_divisors = find_divisors(n_div)
            new_divisors = [i*div for div in old_divisors]
            res = list(set(new_divisors + old_divisors))
            return res
        #
    # If none are found, n is prime, so only 1 and n divide n
    return [1, n]


@numba.njit
def find_first_int_with_target_divsum(target: int, maxfactor=None):
    """Finds the lowest interger n such that the sum of divisors of n is at least the target value.
    If maxfactor is set, only divisors which divide a number at most maxfactor times are included in the sum."""

    if maxfactor is None:
        maxfactor = float("inf")

    n = 1
    while True:
        divs = find_divisors(n)
        sum_ = 0
        for div in divs:
            if n // div <= maxfactor:
                sum_ += div

        if sum_ >= target:
            return n
        n += 1


def solve(data: str):
    n_presents = parse(data)

    target = math.ceil(n_presents / 10)
    star1 = find_first_int_with_target_divsum(target=target)
    print(f"Solution to part 1: {star1}")

    target2 = math.ceil(n_presents / 11)
    star2 = find_first_int_with_target_divsum(target=target2, maxfactor=50)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2015, 20
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
