# ··.. ` + · `.·   +·.* `·  +.·    .`·  .*`·    · `  . ·*· `.  ·+ .· •·. `·. . ·
# `. · ·   *`·  *.` · `· *.   Arithmetic Logic Unit ·  .`·   `*  · .·  +·. · `·*
#  .* ·   ·.·+  `   *· https://adventofcode.com/2021/day/24   ·.· `  ·. *·  `·*·
#  ··`*. · ·* ` ·   .` +· ``.··   .`+ *· .· ·.   .+  ·` ·` ·· . * ·    ·  .`· · 

import functools
import operator
import re
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

# Pattern for the part of the program which processes a single input
block_parts = (
    "inp w",
    "mul x 0",
    "add x z",
    "mod x 26",
    r"div z (?P<z_div>.*)",
    r"add x (?P<offset>.*)",
    "eql x w",
    "eql x 0",
    "mul y 0",
    "add y 25",
    "mul y x",
    "add y 1",
    "mul z y",
    "mul y 0",
    "add y w",
    r"add y (?P<inc>.*)",
    "mul y x",
    "add z y",
)


def parse(s: str) -> list[str]:
    return s.splitlines()


@dataclass
class State:
    """Represents values of the 4 ALU registers."""

    w: int=0
    x: int=0
    y: int=0
    z: int=0


class Emulator:
    """Emulator for the ALU. This isn't really used to solve the problem, I just use
    it to verify solutions."""
    
    # The supported instructions
    ops = dict(
        mul=operator.mul,
        add=operator.add,
        mod=operator.mod,
        div=operator.floordiv,
        eql=lambda a, b: int(a == b),
        inp=lambda _, b: b
    )

    def __init__(self, program: list[str]) -> None:
        self.instructions: list[tuple[str|int, ...]] = []
        for step in program:
            ins = tuple(elem if elem.isalpha() else int(elem) for elem in step.split())
            self.instructions.append(ins)
    
    def run(self, inputs: list[int], initial_state: State|None=None) -> State:
        """Runs the emulator using the specified inputs and initial state (defaults to all zeroes)"""
        
        inputs = list(reversed(inputs))
        state = State() if initial_state is None else initial_state
        
        for ins in self.instructions:
            if ins[0] == "inp":
                # If input instruction, use the next available input
                op, reg = ins
                arg = inputs.pop()
            else:
                op, reg, arg_raw = ins
                arg = arg_raw if isinstance(arg_raw, int) else getattr(state, arg_raw)

            assert isinstance(reg, str) and isinstance(op, str)
            # Look up the value of the first argument, and compute result
            reg_val = getattr(state, reg)
            func = self.ops[op]
            res = func(reg_val, arg)
            # Compute the resulting state
            state = replace(state, **{reg: res})

        assert not inputs
        return state

    def validate(self, model_number: list[int]|int) -> bool:
        """Validates a model number"""
        inputs = [int(digit) for digit in str(model_number)] if isinstance(model_number, int) else model_number
        final_state = self.run(inputs=inputs)
        return final_state.z == 0
    #


@dataclass
class Component:
    """Represents one of the recurring components in the program, which does this:
        increment = z%26 + offset != w
        z = z // z_div
        if increment:
            z = 26z + w + inc
        
        but with a numpy arrays. The result (for given values of z and w) takes a 'low' value if
        z%26 + offset == w and a 'high' value otherwise. This means that z // z_div provides a lower
        bound on the output of any component. This fact can be used for pruning z values which are so large
        that even getting the low output from all subsequent components, will not reach z=0.
    """

    z_div: int
    offset: int
    inc: int

    def __call__(self, z: NDArray[np.int_], w: NDArray[np.int_]) -> NDArray[np.int_]:
        inc_mask = (z % 26) + self.offset != w
        res = z // self.z_div

        res[inc_mask] = 26*res[inc_mask] + w[inc_mask] + self.inc
        return res

    def __post_init__(self) -> None:
        if self.inc < 0:
            raise ValueError
        #
    #


def split_into_components(instructions: list[str]) -> list[Component]:
    """Splits instructions into blocks (each starting with an input), and instantiates
    a compoenent representing each such block."""
    
    # Partition into blocks of instructions, starting with each input
    cutinds = [i for i, line in enumerate(instructions) if line.startswith("inp")] + [len(instructions)]
    blocks = ["\n".join(instructions[a:b]) for a, b in zip(cutinds, cutinds[1:], strict=False)]
    assert len(set(block.count("\n") for block in blocks)) == 1

    res = []
    pattern = re.compile("\n".join(block_parts), flags=re.MULTILINE)

    for block in blocks:
        # Match regex against the code block to determine the component's parameters
        match = pattern.search(block)
        if match is None:
            raise RuntimeError("Couldn't locate pattern for optimization")
        
        # Use the parameters to create a component
        d_s = match.groupdict()
        d = {k: int(v) for k, v in d_s.items()}
        component = Component(**d)
        res.append(component)

    return res


def determine_model_number(components: list[Component], keep_min=False):
    """Determines a model number which passes the validation check.
    components: List of components which perform the validation step for each input.
    keep_min: Whether to keep the minimum input leading to each z value.
        If False (default), the highest valid model number will be returned, and vice versa."""
    
    # Ceiling for z. Each component can divide by some number, so value above their product can be pruned
    z_max = functools.reduce(operator.mul, (c.z_div for c in components))
    dtype = np.int64

    # Array for each current z value, and the input which resulted in that value
    prev = np.array([[0, 0]], dtype=dtype)
    inputs = np.array(list(range(1, 10)), dtype=dtype)

    for component in components:
        # Update ceiling to reflect the remaining components
        z_max //= component.z_div

        # This will hold columns: z (previous), w (previous), w (new input), z (new)
        computations = np.empty(shape=(len(prev)*len(inputs), 4), dtype=dtype)

        # Set all combinations of new inputs and previous z and inputs
        computations[:, 0] = np.tile(prev[:, 0], len(inputs))
        computations[:, 1] = np.tile(prev[:, 1], len(inputs))

        # Set all combinations of new inputs, in the desired order (min/max first)
        new_inputs = np.repeat(inputs, len(prev))
        if not keep_min:
            new_inputs = np.flip(new_inputs)
        computations[:, 2] = new_inputs
        
        # Compute the output z values from the current component, for each z and input
        z_ = computations[:, 0]
        w_ = computations[:, 2]
        computations[:, 3] = component(z_, w_)
        z_new = computations[:, 3]

        # Prune z values that are too large
        in_bounds = computations[z_new <= z_max]
        
        # Get index of first occurrence of each z value. This has the desired input value due to the ordering
        _, inds = np.unique(in_bounds[:, 3], return_index=True)
        keep = in_bounds[inds]

        # Keep each distinct z value and the corresponding best input for next component
        prev = np.empty(shape=(len(keep), 2), dtype=dtype)
        prev[:, 0] = keep[:, 3]
        # Shift previous inputs left and update model numbers
        prev[:, 1] = keep[:, 1]*10 + keep[:, 2]
    
    valid = prev[prev[:, 0] == 0][:, 1]
    assert len(valid) == 1
    res = int(valid[0])
    return res


def solve(data: str) -> tuple[int|str, ...]:
    instructions = parse(data)
    emulator = Emulator(instructions)
    components = split_into_components(instructions)

    star1 = determine_model_number(components)
    assert emulator.validate(star1)    
    print(f"Solution to part 1: {star1}")

    star2 = determine_model_number(components, keep_min=True)
    assert emulator.validate(star2)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 24
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
