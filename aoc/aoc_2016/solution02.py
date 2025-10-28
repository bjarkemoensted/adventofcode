# .··.` + ·  ·`  · .  ·`      ··+ .` •·  .+·`·  ·+` ·  * .  `  ·.   · ` ·.·` *.·
# `.* ·  · * ` ·  ··  `·.       Bathroom Security  + `·  ·. ·`  ·    · * · + ·`.
#  +. · `  .· · *``•·  https://adventofcode.com/2016/day/2 ·  · +``·   `   ·.`· 
# •· `+·.` ·*  ` .·  ·  + ` .·   *`·.  · `     · .·`+  ·  ` •· `  ·+    ·` .·  `


def parse(s: str):
    lines = [line.strip() for line in s.split("\n")]
    return lines


KEYPAD = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]


KEYPAD2 = [
    [None, None, 1,   None, None],
    [None, 2,    3,   4,    None],
    [5,    6,    7,   8,    9],
    [None, "A",  "B", "C",  None],
    [None, None, "D", None, None]
]


def _valid_coord(keypad, coord):
    i, j = coord
    if any(c < 0 for c in coord):
        return False
    try:
        val = keypad[i][j]
        return val is not None
    except IndexError:
        return False


def update_coords(initial, shift, keypad):
    i, j = initial
    if shift == "U":
        i += -1
    elif shift == "D":
        i += 1
    elif shift == "R":
        j += 1
    elif shift == "L":
        j += -1

    updated = (i, j)

    if _valid_coord(keypad, updated):
        return updated
    else:
        return initial


def beepboop(instructions, keypad, starting_loc):
    """Punch in the instructions on the keypad"""
    code = []
    loc = starting_loc
    for sequence in instructions:
        for letter in sequence:
            loc = update_coords(loc, keypad=keypad, shift=letter)
        i, j = loc
        code.append(keypad[i][j])

    return code


def solve(data: str) -> tuple[int|str, int|str]:
    parsed = parse(data)
    
    code = beepboop(parsed, keypad=KEYPAD, starting_loc=(1, 1))
    star1 = ''.join(map(str, code))
    print(f"Solution to part 1: {star1}")

    code2 = beepboop(parsed, keypad=KEYPAD2, starting_loc=(2, 0))
    star2 = ''.join(map(str, code2))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2016, 2
    from aocd import get_data
    raw = get_data(year=year, day=day)


if __name__ == '__main__':
    main()
