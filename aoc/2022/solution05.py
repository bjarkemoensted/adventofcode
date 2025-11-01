import re
from copy import deepcopy


def read_input():
    with open("input05.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    parts = s.split("\n\n")

    # parse the initial setup of the cargo stacks
    raw = [list(line) for line in parts[0].split("\n")[::-1]]

    def isint(x):
        """Helper method to determine if a string can be parsed into an int"""
        try:
            _ = int(x)
            return True
        except ValueError:
            return False

    # Determine the index for each stack of boxes in the input data
    ind2stacknumber = {ind: int(sn) for ind, sn in enumerate(raw[0]) if isint(sn)}
    stacks = {sn: [] for sn in ind2stacknumber.values()}
    # Find the letter on each box and add to the corresponding stack
    for stuff in raw[1:]:
        for ind, sn in ind2stacknumber.items():
            box = stuff[ind]
            if box != " ":
                stacks[sn].append(box)

    # parse the moving instructions
    instructions = []
    for line in parts[1].split("\n"):
        m = re.match(r"move (\d+) from (\d+) to (\d+)", line)
        instruction = tuple(int(subs) for subs in m.groups())
        instructions.append(instruction)

    return stacks, instructions


def move_boxes(n_boxes, stack_from, stack_to, stacks):
    """Moves n boxes from one stack to another, by repeatedly popping the top box"""
    for _ in range(n_boxes):
        box = stacks[stack_from].pop()
        stacks[stack_to].append(box)

    return


def move_boxes_in_order(n_boxes, stack_from, stack_to, stacks):
    """Moves n boxes from one stack to another, maintaining order."""
    boxes = stacks[stack_from][-n_boxes:]
    stacks[stack_to] += boxes
    stacks[stack_from] = stacks[stack_from][:-n_boxes]

    return


def main():
    raw = read_input()
    stacks, instructions = parse(raw)
    stacks1 = deepcopy(stacks)

    for instruction in instructions:
        move_boxes(*instruction, stacks1)

    message = "".join([stacks1[sn][-1] for sn in sorted(stacks1.keys())])
    print(f"The top boxes spell {message}.")

    stacks2 = deepcopy(stacks)
    for instruction in instructions:
        move_boxes_in_order(*instruction, stacks2)

    message = "".join([stacks2[sn][-1] for sn in sorted(stacks2.keys())])
    print(f"The top boxes moved with CrateMover9001 spell {message}.")


if __name__ == '__main__':
    main()
