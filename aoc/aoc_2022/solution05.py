# ..·*  `  *`· +   * ·. *    `·  ·*  .`•·  .+·  .*` +·    .* ·*  `·  .`+ • .*··`
# ·`.·  * `·  +.  +`· *     `*•·. Supply Stacks        . ·*`* .    ·* .`*·• ·+`.
# `*+  ··    *.• .·+ ` https://adventofcode.com/2022/day/5  . `  *   *·.   ·.`*`
#   `.· •+  * ·  `.·  + `  *·.  * · ·+ ` ·.+ •    · `+. *   ·  ·*  `.  *·` . • ·

import re
from copy import deepcopy

type stacktype = dict[int, list[str]]
type instype = tuple[int, int, int]


def parse(s: str) -> tuple[stacktype, list[instype]]:
    parts = s.split("\n\n")

    # parse the initial setup of the cargo stacks
    raw = [list(line) for line in parts[0].splitlines()[::-1]]

    def isint(x):
        """Helper method to determine if a string can be parsed into an int"""
        try:
            _ = int(x)
            return True
        except ValueError:
            return False

    # Determine the index for each stack of boxes in the input data
    ind2stacknumber = {ind: int(sn) for ind, sn in enumerate(raw[0]) if isint(sn)}
    stacks: dict[int, list[str]] = {sn: [] for sn in ind2stacknumber.values()}
    # Find the letter on each box and add to the corresponding stack
    for stuff in raw[1:]:
        for ind, sn in ind2stacknumber.items():
            box = stuff[ind]
            if box != " ":
                stacks[sn].append(box)

    # parse the moving instructions
    instructions: list[tuple[int, int, int]] = []
    for line in parts[1].split("\n"):
        m = re.match(r"move (\d+) from (\d+) to (\d+)", line)
        assert m is not None
        a, b, c = map(int, m.groups())
        instructions.append((a, b, c))

    return stacks, instructions


def move_boxes(n_boxes: int, stack_from: int, stack_to: int, stacks: stacktype) -> None:

    """Moves n boxes from one stack to another, by repeatedly popping the top box"""
    for _ in range(n_boxes):
        box = stacks[stack_from].pop()
        stacks[stack_to].append(box)

    return


def move_boxes_in_order(n_boxes: int, stack_from: int, stack_to: int, stacks: stacktype) -> None:
    """Moves n boxes from one stack to another, maintaining order."""
    boxes = stacks[stack_from][-n_boxes:]
    stacks[stack_to] += boxes
    stacks[stack_from] = stacks[stack_from][:-n_boxes]

    return


def get_seat_id(coord: tuple[int, int]) -> int:
    """Returns the ID of a given seat"""
    i, j = coord
    res = i*8 + j
    return res


def determine_remaining_seat_id(seat_coords: list[tuple[int, int]]) -> int:
    all_IDs = {get_seat_id(coord) for coord in seat_coords}

    for i in range(128):
        for j in range(8):
            sid = get_seat_id((i, j))

            seat_is_free = sid not in all_IDs
            neighbours_exist = all(val+sid in all_IDs for val in (1, -1))
            correct = seat_is_free and neighbours_exist
            if correct:
                return sid
            #
        #
    
    raise RuntimeError


def solve(data: str) -> tuple[int|str, ...]:
    stacks, instructions = parse(data)
    stacks2 = deepcopy(stacks)

    for n_boxes, stack_from, stack_to in instructions:
        move_boxes(n_boxes, stack_from, stack_to, stacks)
        move_boxes_in_order(n_boxes, stack_from, stack_to, stacks2)

    star1 = "".join([stacks[sn][-1] for sn in sorted(stacks.keys())])
    print(f"Solution to part 1: {star1}")

    star2 = "".join([stacks2[sn][-1] for sn in sorted(stacks2.keys())])
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 5
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
