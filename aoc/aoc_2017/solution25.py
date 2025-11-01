# `*·  .·+`   ··*`. · +`.`··       ·  .`·   ·  * `  ·•`.  *. ·· `.•·`    ··`*.`·
# ·`*·.·+`  ·  `` · *·     +`. The Halting Problem  . .·* `     · · .+  · `*· ·`
# `·.`·*.·*  ·  . ` ·. https://adventofcode.com/2017/day/25   `· *   ``  · .·+.·
# ···  ` * `· •.·  ·.` ·  *`.  ·  ·*. ·  •·. `* `·.`* ·    ·`    ·.• ·+·.*··`· .


import re
from dataclasses import dataclass
from typing import Literal, Self, TypeAlias, cast, get_args

import numba

dirtype: TypeAlias = Literal["right", "left"]


@dataclass
class Instruction:
    write: int
    move: dirtype
    new_state: str

    @classmethod
    def from_str_dict(cls, d: dict[str, str]) -> Self:
        move = d["move"]
        assert move in get_args(dirtype)
        
        res = cls(
            write=int(d["write"]),
            move=cast(dirtype, move),
            new_state=d["new_state"]
        )
        return res


# type for the instructions at a given state
instype: TypeAlias = dict[int, Instruction]
# type for an entire 'program' - instructions at each state
progtype: TypeAlias = dict[str, instype]


def _parse_init(s):
    """Parses initial state and number of steps to be run"""
    lines = s.splitlines()
    state = re.match(r"Begin in state (\S+).", lines[0]).group(1)
    n_steps = int(re.match(r"Perform a diagnostic checksum after (.*) steps.", lines[1]).group(1))

    return state, n_steps


def _parse_block(s) -> tuple[str, instype]:
    """Parses a block of instructions - the state for which it takes effect, and for each possible value (0, 1)
    what to write, which direction to move, and what the resulting state is."""

    lines = s.splitlines()
    
    m = re.match(r"In state (\S+):", lines[0])
    assert m is not None
    state = m.group(1)

    interpretations = (
        ("write", r"\s*- Write the value (\d+)."),
        ("move", r"\s*- Move one slot to the (right|left)."),
        ("new_state", r"\s*- Continue with state (.*).")
    )

    operations: instype = dict()
    
    cutinds = [i for i, line in enumerate(lines) if line.strip().startswith("If the current value is")]
    cutinds.append(len(lines))
    
    for a, b in (cutinds[i: i+2] for i in range(len(cutinds)-1)):
        snippet = lines[a:b]
        # Match current value
        m = re.match(r"\s*If the current value is (\d+):", snippet[0])
        assert m is not None
        val = int(m.group(1))
        
        params: dict[str, str] = dict()
        for (key, pattern), line in zip(interpretations, snippet[1:], strict=True):
            m = re.match(pattern, line)
            assert m is not None
            params[key] = m.group(1)
        
        ins = Instruction.from_str_dict(params)
        operations[val] = ins

    return state, operations


def parse(s: str) -> tuple[str, int, progtype]:
    init_, *blocks = s.split("\n\n")

    initial_state, n_steps = _parse_init(init_)
    instructions: progtype = dict()

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


def checksum(program: progtype, initial_state: str, n_steps: int):
    dir_: dict[dirtype, int] = {"left": -1, "right": 1}

    # Convert the instructions to a purely numerical format ('A' -> 0, 'B' -> 1, etc)
    letter2ind = {letter: i for i, letter in enumerate(sorted(program.keys()))}
    numins: list[list[list[int]]] = [[] for _ in range(len(program))]

    for state, steps in program.items():
        stateind = letter2ind[state]
        these_steps: list[list[int]] = [[], []]  # Instruction for encountering values of 0 and 1
        for ifval, ins in steps.items():

            update = [
                ins.write,  # The value can be used as-is
                dir_[ins.move],  # Look up the direction
                letter2ind[ins.new_state]  # Use a number instead of letter for new state
            ]
            
            these_steps[ifval] = update
        numins[stateind] = these_steps

    initial_numeric = letter2ind[initial_state]
    
    # Use typed list so numba doesn't act up
    typed_ins = numba.typed.List(numba.typed.List(part) for part in numins)

    # Run the instructions
    res = _run_numeric(typed_ins, initial_numeric, n_steps)

    return res


def solve(data: str) -> tuple[int|str, None]:
    initial_state, n_steps, instructions = parse(data)

    star1 = checksum(instructions, initial_state, n_steps)
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2017, 25
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
