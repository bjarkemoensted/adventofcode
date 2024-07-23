from ast import literal_eval as LE
from collections import defaultdict
import pathlib
import re


def read_input():
    _here = pathlib.Path(__file__).resolve().parent
    fn = _here / "inputs" / "input08.txt"
    with open(fn) as f:
        puzzle_input = f.read()
    return puzzle_input


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
    from aoc.utils.data import check_examples
    check_examples(year=2017, day=8, solver=solve)
    raw = read_input()
    solve(raw)


if __name__ == '__main__':
    main()
