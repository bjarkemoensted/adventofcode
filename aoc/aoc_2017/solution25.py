import numba
import re


def _parse_init(s):
    """Parses initial state and number of steps to be run"""
    lines = s.splitlines()
    state = re.match(r"Begin in state (\S+).", lines[0]).group(1)
    n_steps = int(re.match(r"Perform a diagnostic checksum after (.*) steps.", lines[1]).group(1))

    return state, n_steps


def _parse_block(s):
    """Parses a block of instructions - the state for which it takes effect, and for each possible value (0, 1)
    what to write, which direction to move, and what the resulting state is."""

    lines = s.splitlines()
    state = re.match(r"In state (\S+):", lines[0]).group(1)

    interpretations = {
        "write": r"\s*- Write the value (\d+).",
        "move": r"\s*- Move one slot to the (right|left).",
        "new_state": r"\s*- Continue with state (.*).",
    }

    operations = dict()

    for line in lines:
        m = re.match(r"\s*If the current value is (\d+):", line)
        if m is not None:
            val = int(m.group(1))
            operations[val] = dict()
            continue

        for type_, pattern in interpretations.items():
            m = re.match(pattern, line)
            if m is not None:
                elem = m.group(1)
                try:
                    elem = int(elem)
                except ValueError:
                    pass
                operations[val][type_] = elem
                break
            #
        #

    return state, operations


def parse(s):
    init_, *blocks = s.split("\n\n")

    initial_state, n_steps = _parse_init(init_)
    instructions = dict()

    for block in blocks:
        state, ops = _parse_block(block)
        instructions[state] = ops

    return initial_state, n_steps, instructions


@numba.njit
def _run_numeric(instructions: list, initial_state: int, n_steps: int) -> int:
    # Note where values are set (i.e. have a value of 1)

    set_ = set(
        [int(val) for val in range(0)] # typecasting empty iterator as int so numba knows which datatype to use
    )

    pos = 0
    state = initial_state

    for _ in range(n_steps):
        # Grab the relevant instructions given the current state and value at current position
        current = int(pos in set_)
        write, move, newstate = instructions[state][current]

        # Update value at current position if necessary
        if write == 1 and not current:
            set_.add(pos)
        elif current and write == 0:
            set_.remove(pos)

        # Update state and position
        pos += move
        state = newstate

    res = len(set_)
    return res


def checksum(instructions: list, initial_state: str, n_steps: int):
    dir_ = {"left": -1, "right": 1}

    # Convert the instructions to a purely numerical format ('A' -> 0, 'B' -> 1, etc)

    letter2ind = {letter: i for i, letter in enumerate(sorted(instructions.keys()))}
    numins = [None for _ in range(len(instructions))]

    for state, ins in instructions.items():
        stateind = letter2ind[state]
        this_instruction = [None, None]  # Instruction for encountering values of 0 and 1
        for ifval, condins in ins.items():
            update = [
                condins["write"],  # The value can be used as-is
                dir_[condins["move"]],  # Look up the direction
                letter2ind[condins["new_state"]]  # Use a number instead of letter for new state
            ]

            this_instruction[ifval] = update
        numins[stateind] = this_instruction

    initial_numeric = letter2ind[initial_state]
    # Use typed list so numba doesn't act up
    typed_ins = numba.typed.List(numba.typed.List(part) for part in numins)

    # Run the instructions
    res = _run_numeric(typed_ins, initial_numeric, n_steps)

    return res


def solve(data: str):
    initial_state, n_steps, instructions = parse(data)

    star1 = checksum(instructions, initial_state, n_steps)
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 25
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
