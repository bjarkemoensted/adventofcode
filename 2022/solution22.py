from copy import deepcopy
import numpy as np


def read_input():
    with open("input22.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


test = """        ...#
        .#..
        #...
        ....
...#.......#
........#...
..#....#....
..........#.
        ...#....
        .....#..
        .#......
        ......#.

10R5L5R10L4R5L5"""


def display(M):
    """Prints the map"""
    lines = []
    for row in M:
        lines.append("".join(row))

    print("\n".join(lines))


def parse(s):
    map_, instructions_raw = s.split("\n\n")

    map_ = map_.split("\n")
    rows = len(map_)
    cols = max([len(line) for line in map_])

    M = np.array([[" " for _ in range(cols)] for _ in range(rows)])
    for i, line in enumerate(map_):
        for j, char in enumerate(line):
            M[i, j] = char

    instructions = []
    buffer = ""
    for char in instructions_raw:
        if char in "LR":
            if buffer:
                instructions.append(int(buffer))
                buffer = ""
            instructions.append(char)
        else:
            buffer += char
        #
    if buffer:
        instructions.append(int(buffer))

    return M, instructions


def identify_faces(M):
    d = {}
    counter = 0
    for ilim in range(0, 200, 50):
        for jlim in range(0, 150, 50):
            face = M[ilim:ilim+50, jlim: jlim+50]
            if np.all(face == " "):
                continue
            print(face)


def determine_starting_position(M):
    """Identify the free tile in the top row furthest to the left"""
    rows, cols = M.shape
    top_row = min(i for i in range(rows) if any(char == "." for char in M[i]))
    leftmost_tile = min(j for j, char in enumerate(M[top_row]) if char == ".")

    return top_row, leftmost_tile


def get_initial_state(M):
    """Returns a starting state"""
    state = {
        "pos": determine_starting_position(M),
        "dir_": ">"
    }

    return state


def get_direction_vector(dir_):
    """Return a direction unit vector (tuple) from a direction character (>, <, v, and ^)"""
    d = {
        ">": (0, 1),
        "v": (1, 0),
        "<": (0, -1),
        "^": (-1, 0)
    }
    res = d[dir_]
    return res


def turn_inplace(state, turn_direction):
    """Returns the new direction after turning left (L) or right (R)."""
    directions = list(">v<^")
    assert turn_direction in "LR"
    inc = 1 if turn_direction == "R" else -1
    ind = directions.index(state["dir_"])
    ind = (ind + inc) % len(directions)
    new_dir = directions[ind]
    state["dir_"] = new_dir


def wrap(pos, M):
    """Wraps position around the edge of the map"""
    res = list(pos)
    for ind in range(len(M.shape)):
        limit = M.shape[ind]
        coord = pos[ind]
        if not (0 <= coord < limit):
            coord = coord % limit
        res[ind] = coord

    return tuple(res)


def _step_direction(pos, dir_):
    """Moves pos one step in direction dir_, notwithstanding falling off the map."""
    vec = np.array(pos)
    delta = np.array(get_direction_vector(dir_))

    res = tuple(vec+delta)
    return res


def find_next_free_tile(M, state):
    """Keeps stepping forward until a free tile is found. Returns None if not possible."""
    pos0 = state["pos"]
    dir_ = state["dir_"]

    pos = state["pos"]
    while True:
        pos = _step_direction(pos, dir_)
        pos = wrap(pos, M)
        i, j = pos
        if pos == pos0 or M[i, j] == "#":
            return None
        if M[i, j] in ".><^v":
            return pos


def step_inplace(M, state):
    """Takes a single step in the specified direction, wrapping around edges. Returns None if not possible.
    Updates state in-place."""
    new_pos = find_next_free_tile(M, state)
    if new_pos is None:
        return

    i, j = state["pos"]
    M[i, j] = state["dir_"]
    state["pos"] = new_pos


def follow_instructions(M, state, instructions):
    """Follows list of instructions"""
    M = deepcopy(M)
    state = deepcopy(state)
    for instruction in instructions:
        if isinstance(instruction, int):
            for _ in range(instruction):
                step_inplace(M, state)
        elif instruction in "LR":
            turn_inplace(state, instruction)
        else:
            raise ValueError

    return M, state


def find_password(state):
    """Finds password from the final state, after instructions have been followed."""
    row, col = state["pos"]
    res = 1000*(row + 1) + 4*(col + 1)

    direction_score = list(">v<^")
    res += direction_score.index(state["dir_"])

    return res


def main():
    raw = read_input()
    M, instructions = parse(raw)
    state = get_initial_state(M)

    #_, endstate = follow_instructions(M, state, instructions)
    #password = find_password(endstate)
    #print(f"The final password is {password}.")

    hmm = identify_faces(M)





if __name__ == '__main__':
    main()
