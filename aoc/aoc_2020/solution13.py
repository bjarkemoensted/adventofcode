# .·  •`.    ·*.` .·`* ·     +·         ·*`.· *  ·      ·. ·* .•     ·   .`· * .
# ·`.·  *.` ·  +· · .  +· .   `·  Shuttle Search     *. `·   ·   * · *. +` . ·.·
# ·*` ·     *.·     ·` https://adventofcode.com/2020/day/13  `   · .`  * ·  · *`
# *.· ··  ·+.` · *    ·.   ·+  .` *·   ·  ·      `. ·      +·* .·`  ·  .*    .· 

import functools
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Bus:
    id_: int
    index: int


def parse(s: str) -> tuple[int, list[Bus]]:
    ts_line, bus_line = s.splitlines()
    timestamp = int(ts_line)

    busses = []
    for index, entry in enumerate(bus_line.split(",")):
        if entry != "x":
            bus = Bus(id_=int(entry), index=index)
            busses.append(bus)
        #

    return timestamp, busses


def get_waittime(timestamp: int, bus: Bus):
    """Determines the waiting time for the given bus ID"""
    if timestamp % bus.id_ == 0:
        return 0
    else:
        return bus.id_ - timestamp % bus.id_
    #


def prime_factor(n: int) -> dict[int, int]:
    """Computes the prime factors of the input number.
    Returns a dict, where for each key-value pair k, v
    k is a prime factor, and v is its power, so summing
    k**v will give n."""

    res: dict[int, int] = defaultdict(int)
    running: int = n

    for i in range(2, n + 1):
        while running % i == 0:
            res[i] += 1
            running //= i
        #

    return res


def get_least_common_divisor(*numbers: int) -> int:
    """Returns the least common divisor of the unput numbers.
    This works by prime factoring each number, keeping the largest power
    of each prime factor encountered."""
    factors: dict[int, int] = {}

    # Keep the largest power of each prime factor of the numbers
    for n in numbers:
        d = prime_factor(n)
        for fac, mul in d.items():
            factors[fac] = max(mul, factors.get(fac, 0))
        #

    # Sum the max powers of each prime factor
    contributions = (fac**mul for fac, mul in factors.items())
    final = functools.reduce(lambda a, b: a*b, contributions, 1)
    return final


def iterative_scan(*busses: Bus):
    """Finds the minimum wait time. Starts by incrementing by one, then when another bus
    has been added to the departure 'chain', the increment is changed to the least common divisor
    between all busses in the chain. I think this is a caveman version of the
    Chinese Remainder Theorem"""

    increment = 1
    running = 1
    chain: list[int] = []

    for bus in busses:
        remainder = bus.index

        while get_waittime(running, bus) != remainder % bus.id_:
            running += increment
        
        chain.append(bus.id_)
        increment = get_least_common_divisor(*chain)

    return running


def solve(data: str) -> tuple[int|str, ...]:
    timestamp, busses = parse(data)

    best_route = min(busses, key=lambda bus: get_waittime(timestamp, bus))
    waittime = get_waittime(timestamp, best_route)

    star1 = best_route.id_*waittime
    print(f"Solution to part 1: {star1}")

    star2 = iterative_scan(*busses)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 13
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
