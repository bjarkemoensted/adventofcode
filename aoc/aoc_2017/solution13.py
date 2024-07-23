import pathlib


def read_input():
    _here = pathlib.Path(__file__).resolve().parent
    fn = _here / "inputs" / "input13.txt"
    with open(fn) as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = dict()
    for line in s.splitlines():
        k, v = map(int, line.split(": "))
        res[k] = v
    return res


def _phase(range_: int) -> int:
    """Recurrence time for a scanner moving back and forth between a range"""
    res = 2*range_ - 2
    return res


def iterate_path(phases: dict, delay: int = 0):
    """Provides the depths of the layers where caught"""
    time_ = delay
    pos = 0
    passed = 0
    while passed < len(phases):
        try:
            caught = time_ % phases[pos] == 0
            passed += 1
        except KeyError:
            caught = False
        if caught:
            yield pos
        pos += 1
        time_ += 1
    #


def compute_severity(firewall: dict):
    phases = {depth: _phase(range_) for depth, range_ in firewall.items()}
    res = sum([depth*firewall[depth] for depth in iterate_path(phases)])

    return res


def determine_best_delay(firewall: dict):
    """Brute force approach to finding the earliest time where no scanner catch the package."""
    phases = {depth: _phase(range_) for depth, range_ in firewall.items()}
    delay = 0
    while True:
        try:
            next(iterate_path(phases, delay))
            delay += 1
        except StopIteration:
            return delay


def solve(data: str):
    firewall = parse(data)

    star1 = compute_severity(firewall)
    print(f"Solution to part 1: {star1}")

    star2 = determine_best_delay(firewall)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    from aoc.utils.data import check_examples
    check_examples(year=2017, day=13, solver=solve)
    raw = read_input()
    solve(raw)


if __name__ == '__main__':
    main()
