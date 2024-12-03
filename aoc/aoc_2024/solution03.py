# ⸳ •` `  ⸳  * * `⸳⸳ꞏ .*⸳ *     ꞏ  + .`*⸳ .ꞏ `  • ⸳`. ⸳ ꞏ   + `.⸳    *ꞏ `  .*⸳ꞏ.
#   ⸳.• ⸳`ꞏ     ꞏ*•*`⸳ `ꞏ. ` ⸳     Mull It Over ⸳   • `  .⸳`  * ꞏ+ ꞏ ⸳      `.`+
#  •`*.⸳* `ꞏ ⸳ ⸳•.   ꞏ https://adventofcode.com/2024/day/3    ⸳* .` •   * ` .`⸳ꞏ
# +ꞏ. ⸳* ⸳+ `. ꞏ ꞏ  *`  .⸳`   •   ⸳ ꞏ .  ⸳  ⸳ꞏ * * ⸳`    ꞏ`•     `. ⸳.*   *⸳ꞏ •⸳

import re


def parse(s):
    res = s
    return res


def scan(s, include_conditionals=False):
    """Scans across the input, looking for multiplication instructions and do's and don't's."""

    commands = []  # Holds numbers for multiplications
    pattern = r"^mul\((\d{1,3}),(\d{1,3})\)"
    
    # Stuff for determining 'state' e.g. whether mul-instructions encountered are accepted or not
    active = True
    enable = "do()"
    disable = "don't()"        
    
    pos = 0
    
    while pos < len(s):
        remainder = s[pos:]
        
        # If current substring matches state-altering strings (do/dont), update and skip ahead
        if include_conditionals:
            if remainder.startswith(enable):
                active = True
                pos += len(enable)
                continue
            elif remainder.startswith(disable):
                active = False
                pos += len(disable)
                continue
            #
        
        # If active and matching a mul-instruction, add to result
        if active:
            m = re.match(pattern, remainder)
            if m:
                nums = tuple(map(int, m.groups()))
                commands.append(nums)
                pos += len(m.group())
                continue
            #
        
        # If not match, move to the next character
        pos += 1
    
    return commands


def solve(data: str):
    parsed = parse(data)
    commands = scan(parsed)

    star1 = sum(a*b for a, b in commands)
    print(f"Solution to part 1: {star1}")

    new_commands = scan(parsed, include_conditionals=True)
    star2 = sum(a*b for a, b in new_commands)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2024, 3
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
