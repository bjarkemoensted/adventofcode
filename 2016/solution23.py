_test = """cpy 2 a
tgl a
tgl a
tgl a
cpy 1 a
dec a
dec a"""


def read_input():
    #return _test
    with open("input23.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def _toggle(instruction):
    """Toggles a given instruction.
    inc -> dec, others -> inc for single-argument instructions,
    jnz -> cpy, others -> jnz for double-argument instructions"""

    fun, *args = instruction
    newfun = None
    if len(args) == 1:
        if fun == "inc":
            newfun = "dec"
        else:
            newfun = "inc"
        #
    elif len(args) == 2:
        if fun == "jnz":
            newfun = "cpy"
        else:
            newfun = "jnz"
    else:
        raise ValueError

    res = (newfun, *args)
    return res


def parse(s):
    res = []
    for line in s.split("\n"):
        ins = line.strip().split()
        for i in range(len(ins)):
            try:
                ins[i] = int(ins[i])
            except ValueError:
                pass
            #
        res.append(tuple(ins))
    return res


# The sequence of instructions and instructions lengths (1 + n_args) required to spot a multiplication instruction
_multiplication_instructions = (('cpy', 3), ('inc', 2), ('dec', 2), ('jnz', 3), ('dec', 2), ('jnz', 3))


def detect_mul(instructions):
    """
    Spot stuff like:
    cpy b c
    inc a
    dec c
    jnz c -2
    dec d
    jnz d -5

    The above multiplies values b and d and add them to register a.
    An equivalent instruction is then
    mul b, d, a (a += b*d)
    (followed by a number of NOPs to leave the indices of other instructions intact)

    This method the equivalent multiplication instruction, or None of no such instruction exists.
    """

    # Can't replace if the number and signatures of instructions are incorrect
    if len(instructions) != len(_multiplication_instructions):
        return
    for ins, (type_, len_) in zip(instructions, _multiplication_instructions):
        if ins[0] != type_ or len(ins) != len_:
            return
        #

    # Check that the instructions can be interpreted as multiplication
    b, c = instructions[0][1:]
    a = instructions[1][1]
    d = instructions[-1][1]
    required = ((2, 1, c), (3, 1, c), (3, 2, -2), (4, 1, d), (5, 1, d), (5, 2, -5))
    if any(instructions[i][j] != req for i, j, req in required):
        return

    res = ("mul", b, d, a)
    return res


def peephole_optimize(instructions):
    """Takes a list of instructions. Returns an optimized version where repeated additions have been replaced
    by multiplication instructions."""
    instructions = [ins for ins in instructions]
    n_ins = len(_multiplication_instructions)

    i = 0
    while i < len(instructions):
        # Check for multiplication in the current window
        window_end = i+n_ins
        multiply_instruction = detect_mul(instructions[i:window_end])
        if multiply_instruction is None:
            i += 1
            continue
        # Found a multiplication optimization. Replace the first instruction by multiplication
        instructions[i] = multiply_instruction
        i += 1
        while i < window_end:
            # Replace remaining instructions in the window with NOPs
            instructions[i] = ("nop",)
            i += 1

    return instructions


class Registry:
    _key_type = str

    def __init__(self, keys=None, **kwargs):
        """Registry class for holding data and handling setting and getting data from registers.
        Starting values of registers can be specified as keyword arguments."""

        if keys is None:
            keys = ("a", "b", "c", "d")
        if not all(isinstance(k, self._key_type) for k in keys):
            raise TypeError

        self._keys = keys  # store ordered keys for displaying purposes
        self._vals = {k: kwargs.get(k, 0) for k in keys}  # Set registers to zero unless values are provided

    def __getitem__(self, item):
        """This allows reg[42] to give while reg['a'] gives the value of register a."""
        if isinstance(item, self._key_type):
            return self._vals[item]
        return item

    def __setitem__(self, key, value):
        """Allows setting registers using values or references to other registers"""
        if not isinstance(key, self._key_type):
            raise TypeError(f"{key} is not type {self._key_type}")

        new_val = self[value]
        self._vals[key] = new_val

    def __str__(self):
        parts = (f"{k} = {self._vals[k]}" for k in self._keys)
        s = ", ".join(parts)
        return s


class Interpreter:
    def __init__(self, registry: Registry, instructions, verbose=False):
        """Optimized interpreter which handles multiplication."""

        self.registry = registry

        # Store the 'raw' and optimized instructions to make wure toggle instructions don't mess things up
        self._base_instructions = [ins for ins in instructions]
        self.instructions = peephole_optimize(self._base_instructions)
        self.pointer = 0

        # Stuff for debugging and spotting which instructions are run the most
        self.verbose = verbose
        self.n_its = 0
        self.counts = [0 for _ in self.instructions]

    def _debug(self):
        """Prints details about current instruction and registry values"""
        ins = self.instructions[self.pointer]
        n = self.counts[self.pointer]
        s = f"{self.n_its}: Instruction {self.pointer}: {ins} ({self.registry}). n_calls: {n}."
        print(s)

    def run_single(self):
        """Runs a single instruction"""

        self.counts[self.pointer] += 1
        instruction = self.instructions[self.pointer]
        if self.verbose:
            self._debug()

        ins, *args = instruction
        fun = getattr(self, ins)
        fun(*args)
        self.n_its += 1

    def run_instructions(self):
        while 0 <= self.pointer < len(self.instructions):
            self.run_single()
        #

    def cpy(self, x, y):
        self.registry[y] = x
        self.pointer += 1

    def inc(self, x):
        self.registry[x] += 1
        self.pointer += 1

    def dec(self, x):
        self.registry[x] -= 1
        self.pointer += 1

    def jnz(self, x, y):
        if self.registry[x] == 0:
            self.pointer += 1
        else:
            self.pointer += self.registry[y]
        #

    def tgl(self, x):
        """Runs a toggle instruction which modifies the instructions"""
        shift = self.registry[x]
        target_ind = self.pointer + shift

        # Ignore instruction if trying to modify an instruction that doesn't exist (out of bounds)
        if 0 <= target_ind < len(self._base_instructions):
            # Modify the 'raw' instructions
            target_ins = self._base_instructions[target_ind]
            new_ins = _toggle(target_ins)
            self._base_instructions[target_ind] = new_ins
            # Rerun optimization in case anything changed
            self.instructions = peephole_optimize(self._base_instructions)

        self.pointer += 1

    def mul(self, x, y, z):
        self.registry[z] += self.registry[x]*self.registry[y]
        self.pointer += 1

    def nop(self):
        self.pointer += 1


def main():
    raw = read_input()
    instructions = parse(raw)
    int_ = Interpreter(registry=Registry(a=7), instructions=instructions, verbose=False)
    int_.run_instructions()

    star1 = int_.registry["a"]
    print(f"After running the instructions, register a contains the value {star1}.")

    int2 = Interpreter(registry=Registry(a=12), instructions=instructions, verbose=False)
    int2.run_instructions()
    star2 = int2.registry["a"]
    print(f"After running the instructions, register a contains the value {star2}.")


if __name__ == '__main__':
    main()
