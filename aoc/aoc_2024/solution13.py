# .··.* ·`    ·*   · .  · * ·  .`··. *·. · `.·+   ·*·.  · `*·+·   ·` . ·* +·` .·
# ·  •`. *.· ·   ·.*·    *··`·   Claw Contraption ..  · ·  `+·.  ·. ··`  *· .+`·
# ·.*`   ·  ·•` ·*  .  https://adventofcode.com/2024/day/13   `· .•·   · · .*· `
# •`.· ·` ·.*· · .`.+* ·       · · . · *·`+.  · . ·+`· .   ·   * `· .   ·. *·`·.


import re
from scipy import optimize


def parse(s: str):
    """Parses into a list of dicts, each with keys "A", "B", and "prize". Each maps to a tuple of ints representing
    x-y coordinates. For A/B that's the x, and y-components of a claw movement resulting from pressing the button.
    For the 'prize key', it means the coordinates of the target location."""
    
    # Regexes to match button/prize parts
    button_pattern = re.compile(r"Button (?P<button>\S): X(?P<x>[+-]\d*), Y(?P<y>[+-]\d*)")
    prize_pattern = re.compile(r"Prize: X=(?P<x>\d*), Y=(?P<y>\d*)")
    
    def xy_from_match(m):
        """Grabs a tuple of integers of x/y groups in a match object"""
        t = tuple(int(m.group(v)) for v in ("x", "y"))
        return t
    
    res = []
    
    for snippet in s.split("\n\n"):
        d = dict()
        parts = snippet.splitlines()
        
        # Match the buttons
        buttons = parts[:2]
        for b in buttons:
            m = re.match(button_pattern, b)
            btn = m.group("button")
            d[btn] = xy_from_match(m)
        
        # Match the prize part
        m = re.match(prize_pattern, parts[-1])
        d["prize"] = xy_from_match(m)
        
        res.append(d)

    return res


def optimize_claw_machine(d: dict, a_cost=3, b_cost=1, target_offset=0):
    """Determines the lowest possible cost of obtaining the prize from the input claw machine.
    Returns None if the prize cannot be obtained."""
    
    # Set up the optimization problem
    c = [a_cost, b_cost]
    bts = ("A", "B")
    b = [crd+target_offset for crd in d["prize"]]
    A = [[d[k][i] for k in bts] for i in range(len(b))]
    
    res = optimize.linprog(
        c,
        A_eq=A,
        b_eq=b,
        integrality=[1, 1],  # require integer solutions
        options={
            "disp": False,
            "presolve": False,  # presolving causes some issues with large input, it seems
        },
        method="highs",
    )

    if res.x is None:
        return
    
    # Verify that the claw ends up at the target
    target = tuple(sum(round(c_t*c_s) for c_t, c_s in zip(res.x, row)) for row in A)
    assert all(a == b for a, b in zip(target, b, strict=True))
    
    # Compute the cost and return it
    res = sum(cost*round(n) for cost, n in zip(c, res.x))
    return res


def get_lowest_total_cost(claw_machines: list, target_offset=0):
    """Computes the lowest possible price (number of tokens) to obtain all possible prizes from the claw machines."""
    
    res = 0
    for d in claw_machines:
        cost = optimize_claw_machine(d, target_offset=target_offset)
        # Add the cost if the prize can be obtained
        if cost is not None:
            res += cost
        #
    
    return res


def solve(data: str) -> tuple[int|str, int|str]:
    claw_machine_data = parse(data)
    
    star1 = get_lowest_total_cost(claw_machine_data)
    print(f"Solution to part 1: {star1}")

    offset = 10000000000000
    star2 = get_lowest_total_cost(claw_machine_data, target_offset=offset)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 13
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()