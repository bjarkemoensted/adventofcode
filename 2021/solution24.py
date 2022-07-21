
example1 = \
"""inp x
mul x -1"""

example2 = \
"""inp z
inp x
mul z 3
eql z x"""

example3 = \
"""inp w
add z w
mod z 2
div w 2
add y w
mod y 2
div w 2
add x w
mod x 2
div w 2
mod w 2"""


with open("input24.txt") as f:
    puzzle_input = f.read()


class State:
    def __init__(self, values=None):
        if values is None:
            values = {char: 0 for char in "wxyz"}
        else:
            values = dict(values)
        self.values = values

    def __getitem__(self, key):
        try:
            val = self.values[key]
        except KeyError:
            val = int(key)
        return val

    def __setitem__(self, key, value):
        self.values[key] = self[value]

    def __str__(self):
        lines = [f"{k}={v}" for k, v in sorted(self.values.items())]
        res = ", ".join(lines)
        return res

    def as_tuple(self):
        tup = tuple(sorted(self.values.items()))
        return tup


class ALU:
    def set_state(self, state):
        if isinstance(state, State):
            self.state = state
        else:
            self.state = State(state)

    def get_state(self):
        return self.state.as_tuple()

    def __init__(self, initial_state=None):
        state = initial_state if isinstance(initial_state, State) else State(initial_state)
        self.state = state

    def inp(self, a, b):
        self.state[a] = b

    def add(self, a, b):
        self.state[a] = self.state[a] + self.state[b]

    def mul(self, a, b):
        self.state[a] = self.state[a] * self.state[b]

    def eql(self, a, b):
        self.state[a] = int(self.state[a] == self.state[b])

    def mod(self, a, b):
        invalid_input = self.state[a] < 0 or self.state[b] <= 0
        if invalid_input:
            raise ValueError

        self.state[a] = self.state[a] % self.state[b]

    def div(self, a, b):
        if self.state[b] == 0:
            raise ZeroDivisionError
        self.state[a] = self.state[a] // self.state[b]

    def __str__(self):
        return self.state.__str__()

    def run_instructions(self, instructions, inputs):
        if isinstance(inputs, str):
            inputs = [int(c) for c in inputs]

        n_inputs = sum(instruction[0] == "inp" for instruction in instructions)
        if len(inputs) != n_inputs:
            raise ValueError(f"Number of inputs ({len(inputs)}) doesn't match number of inp operations ({n_inputs}).")

        for instruction in instructions:
            op = instruction[0]
            fun = getattr(self, op)
            args = instruction[1:]
            if op == "inp":
                args += [inputs.pop(0)]

            fun(*args)


def parse_instructions(s):
    instructions = []
    for line in s.split("\n"):
        instruction = line.strip().split()
        instructions.append(instruction)
    return instructions


def breakdown_instructions_into_blocks(instructions):
    blocks = []
    current = []
    for i, instruction in enumerate(instructions):
        op = instruction[0]
        if op == "inp" and current:
            blocks.append(current)
            current = [instruction]
        else:
            current.append(instruction)
        #
    if current:
        blocks.append(current)

    return blocks



instructions_puzzle = parse_instructions(puzzle_input)
alu = ALU()
initial_state = alu.get_state()

blocks = breakdown_instructions_into_blocks(instructions_puzzle)


all_digits = "123456789"


def crack(minimize=False):
    """Determines the maximum (or minimum) valid model number.
    Only values of z are carried over between each instruction block, so we maintain info on the values of
    z it is possible to obtain as output for each block, and the highest/lowest intermediary code resulting in that
    value."""

    z2best_code = {0: ""}
    intermediate_blocks = blocks[:-1]
    final_block = blocks[-1]
    # Go over the first N-1 instruction blocks.
    for i, block in enumerate(intermediate_blocks):
        newstates = {}  # Stores the output values of z from this instruction block

        # Stuff for printing status
        n_checks = len(all_digits) * len(z2best_code)
        n_checked = 0
        base_msg = f"Iteration {i + 1} of {len(intermediate_blocks)}. Observing {len(z2best_code)} z-values."

        # If we minimze, we go over digits low to high, otherwise high to low.
        digits_list = list(all_digits)
        if minimize:
            digits_list.reverse()

        # Append all digits to all intermediary codes and register the resulting values for z.
        for digit in all_digits:
            for oldz, oldcode in z2best_code.items():

                d = dict(w=0, x=0, y=0, z=oldz)
                state = State(d)

                alu.set_state(state)
                alu.run_instructions(block, digit)
                newz = alu.state['z']
                newcode = oldcode + digit
                # If new code is better than the current, or if z value if new, store the result
                try:
                    current_code = newstates[newz]
                    keep_this = newcode < current_code if minimize else newcode > current_code
                    if keep_this:
                        newstates[newz] = newcode
                except KeyError:
                    newstates[newz] = newcode

                # Print status
                n_checked += 1
                if n_checked % 10000 == 0:
                    msg = f"Checked {100 * n_checked / n_checks:.1f}% of combinations."
                    print(base_msg + " " + msg, end="\r")
                #
            #
        # Update the intermediary codes to contain z values for the current instruction block
        z2best_code = newstates

    print()

    # We now have z values and codes at the final block. Return the first valid code we encounter.
    n_final = 0
    n_states = len(z2best_code)
    for z, code in sorted(z2best_code.items(), key=lambda t: t[1], reverse=not minimize):
        for digit in sorted(all_digits, reverse=not minimize):
            n_final += 1
            if n_final % 10000 == 0:
                print(f"Evaluating final block: {n_final/n_states:.1f}%.", end="\r")
            d = dict(w=0, x=0, y=0, z=z)
            state = State(d)
            alu.set_state(state)
            alu.run_instructions(final_block, digit)
            if alu.state["z"] == 0:
                print()
                res = code + digit
                return res


highest = crack()
print(f"*** Highest possible model number is: {highest} ***")
print()

lowest = crack(minimize=True)
print(f"*** Lowest possible model number is: {lowest} ***")
