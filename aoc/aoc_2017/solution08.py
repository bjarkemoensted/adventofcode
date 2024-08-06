from collections import defaultdict
import re


def parse(s):
    pat = (r"(?P<register>\S+) (?P<operation>\S+) (?P<value>-?\d+) "
           r"if (?P<if_register>\S+) (?P<if_relation>\S+) (?P<if_value>-?\d+)")

    numbers = ("value", "if_value")
    keys = ('register', 'operation', 'value', 'if_register', 'if_relation', 'if_value')
    res = []
    for line in s.splitlines():
        m = re.match(pat, line)
        d = {k: int(v) if k in numbers else v for k, v in m.groupdict().items()}
        ins = tuple(int(d[k]) if k in numbers else d[k] for k in keys)
        res.append(ins)

    return res


_comparisons = {
    '==' : 'eq',
    '>' : 'gt',
    '<' : 'lt',
    '!=': 'ne',
    '>=': 'ge',
    '<=': 'le'
}


def solve(data: str):
    instructions = parse(data)
    verbose = False

    highest_val = 0
    registers = defaultdict(lambda: 0)
    for register, operation, value, if_register, if_relation, if_value in instructions:
        comparison = f"__{_comparisons[if_relation]}__"
        comparison_method = getattr(registers[if_register], comparison)
        condition = comparison_method(if_value)

        val = -1 * value if operation == 'dec' else value

        if verbose:
            cond = f"{registers[if_register]} {if_relation} {if_value}"
            print(cond, condition)

        if condition:
            registers[register] += val
            highest_val = max(highest_val, registers[register])

    star1 = max(registers.values())
    print(f"Solution to part 1: {star1}")

    star2 = highest_val
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 8
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
