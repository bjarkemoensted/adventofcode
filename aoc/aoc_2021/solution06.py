# *·` ·  .·  * ` · *   ·    +· `  ·*`    ··`   · * .` • ·.·   * ·  +  ·.*`    · 
# ·.·``· *.     ·`  •·   ·. `*  ·  Lanternfish . ` + ··`  *·` · •  .   ·`*· · .`
# ·* .  ··• `. ·    ·. https://adventofcode.com/2021/day/6 *  • .*·   `  ·•+ .`·
# ` .·*`+   .··`. •·  · `+  · ·*+ `     ·. +  `*·.  ·   *` .·* ·     ·.   `.*·· 

from collections import Counter


def parse(s: str) -> dict[int, int]:
    fish_countdowns = [int(x) for x in s.strip().split(",")]
    res = dict(Counter(fish_countdowns))
    return res


def run_simulation(fish: dict[int, int], n_iterations=80) -> int:
    """Simulates the lanternfish for the input number of iterations.
    Returns the total."""
    
    fish = {k: v for k, v in fish.items()}

    for i in range(n_iterations):
        # Number of new fish spawned in this round
        n_new = fish.get(0, 0)

        # Decrement the counter until new offspring for all fish
        new_data = {i: 0 for i in range(9)}
        for k, v in fish.items():
            # If fish spawn (timer=0), timer resets to 6. Else decrease by 1.
            new_key = 6 if k == 0 else k - 1
            new_data[new_key] += v

        # Add new fish to state
        new_data[8] += n_new
        fish = new_data

    return sum(fish.values())


def solve(data: str) -> tuple[int|str, ...]:
    fish_data = parse(data)

    star1 = run_simulation(fish_data)
    print(f"Solution to part 1: {star1}")

    star2 = run_simulation(fish_data, n_iterations=256)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 6
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
