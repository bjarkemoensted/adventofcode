# .·    ·*.·+`*·.·`      ·    ·+ `·.·*     • ·. ·`  *.`·*    ` ·    .` ·* . ` ··
# `+·.   · *··.`+ ·`  . The Tyranny of the Rocket Equation .  ` ·  ·* .`··`   *.
# ·.` ·  .`    · ·* ·· https://adventofcode.com/2019/day/1  ·  *  ·.·   +. `··`+
# ` +·. ·`· * `.·   ` · *.` · .*·     ·. ·`   * +·.·     .`·  +  · * ·  `* ·  .`


def parse(s: str) -> list[int]:
    res = list(map(int, s.splitlines()))
    return res


def fuel_requirement(mass: int, include_fuel_for_fuel=False) -> int:
    """Compute fuel required for the specified mass. If include_fuel_for_fuel,
    the fuel required for the additional fuel is included, recursively"""
    
    res = max(0, (mass // 3) - 2)
    if include_fuel_for_fuel and res > 0:
        res += fuel_requirement(res, include_fuel_for_fuel)
    
    return res


def solve(data: str) -> tuple[int|str, ...]:
    modules = parse(data)

    star1 = sum(fuel_requirement(module) for module in modules)
    print(f"Solution to part 1: {star1}")

    star2 = sum(fuel_requirement(module, True) for module in modules)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 1
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
