# •` ·  .·* *`  · ·  .`*·    * .  `·    `  · .  `·• ` ·* . ·+·`   ·.    · ` . *·
# .·*  ·   ·`.·* . ·+  `*   I Heard You Like Registers .·`   +· ·  `.  •.` ·.` ·
# ·`· `.   *.·`·  .+·  https://adventofcode.com/2017/day/8     *`·.·*    · ` .· 
# . `* `·.· ·*  +·   ·.  •· `·   ·*  . ` ·`     *·..·+  `· * .      `*· `. ·+· .


import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Instruction:
    register: str
    operation: str
    value: int
    if_register: str
    if_relation: str
    if_value: int


def parse(s: str) -> list[Instruction]:
    pat = (r"(?P<register>\S+) (?P<operation>\S+) (?P<value>-?\d+) "
           r"if (?P<if_register>\S+) (?P<if_relation>\S+) (?P<if_value>-?\d+)")

    res = []
    for line in s.splitlines():
        m = re.match(pat, line)
        assert m is not None
        
        d = m.groupdict()
        ins = Instruction(
            d["register"],
            d["operation"],
            int(d["value"]),
            d["if_register"],
            d["if_relation"],
            int(d["if_value"]),
        )
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


def solve(data: str) -> tuple[int|str, int|str]:
    instructions = parse(data)
    verbose = False

    highest_val = 0
    registers: dict[str, int] = defaultdict(lambda: 0)
        
    for ins in instructions:
        comparison = f"__{_comparisons[ins.if_relation]}__"
        comparison_method = getattr(registers[ins.if_register], comparison)
        condition = comparison_method(ins.if_value)
        val = -1 * ins.value if ins.operation == 'dec' else ins.value

        if verbose:
            cond = f"{registers[ins.if_register]} {ins.if_relation} {ins.if_value}"
            
            print(cond, condition)

        if condition:
            registers[ins.register] += val
            highest_val = max(highest_val, registers[ins.register])

    star1 = max(registers.values())
    print(f"Solution to part 1: {star1}")

    star2 = highest_val
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2017, 8
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
