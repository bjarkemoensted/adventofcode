# ⸳  ꞏ•*ꞏ . `ꞏ *⸳`* .  ⸳ * .+  ⸳ ` +.      ꞏ. *` ꞏ⸳ `   +  `⸳ •⸳* .     ⸳*ꞏ.   ꞏ
# ` .+⸳ `ꞏ• * ⸳       Perfectly Spherical Houses in a Vacuum .ꞏ     ꞏ+`   + .+ `
# `* .ꞏ⸳     `ꞏ    +   https://adventofcode.com/2015/day/3 .•  `    .   +.`⸳ ꞏ*.
# ꞏ.ꞏ`  *.⸳      * .` •.`⸳*  ꞏ ꞏ*  .`⸳+.  ꞏ* `⸳  .  +•.  ꞏ⸳+`       * ꞏ⸳.  ` *ꞏ+


def parse(s):
    res = list(s)
    return res


def update_coords(coords, instruction):
    x, y = coords
    if instruction == "^":
        y += 1
    elif instruction == "v":
        y -= 1
    elif instruction == ">":
        x += 1
    elif instruction == "<":
        x -= 1
    else:
        raise ValueError

    res = (x, y)
    return res


def compute_houses_with_presents(instructions, startat=(0, 0)):
    current_pos = startat
    pos2n_presents = {current_pos: 1}

    for instruction in instructions:
        current_pos = update_coords(current_pos, instruction)
        pos2n_presents[current_pos] = pos2n_presents.get(current_pos, 0) + 1

    return len(pos2n_presents)


def compute_n_houses_with_presents_using_robot_reindeer(instructions, startat=(0, 0)):
    current_positions = [startat, startat]
    present_data = [{startat: 1} for pos in current_positions]
    for i, instruction in enumerate(instructions):
        ind = i % 2

        current_positions[ind] = update_coords(current_positions[ind], instruction)
        d = present_data[ind]
        d[current_positions[ind]] = d.get(current_positions[ind], 0) + 1

    santa, robo_santa = present_data
    distinct_houses = set(santa.keys()).union(robo_santa.keys())
    return len(distinct_houses)


def solve(data: str):
    instructions = parse(data)

    star1 = compute_houses_with_presents(instructions)
    print(f"Solution to part 1: {star1}")

    star2 = compute_n_houses_with_presents_using_robot_reindeer(instructions)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2015, 3
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
