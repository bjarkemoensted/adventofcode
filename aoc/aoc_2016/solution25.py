# *·.`·  *  · .` ·    •  · ·.*   `·  +     *.`·     .*·   `·*·.    .·•`   ·  ``*
# . · ` ·`*· •·.   ·       `*·. ·  Clock Signal ·+` ·.   ·    `·  · `   .· *`· `
# ·*`. *`·   ·  ·.  ·` https://adventofcode.com/2016/day/25   • · `* ·. * `· ·.·
# *·  ·` .· `     · *` ·   .· ·+` ·   .*+.` ·     ·      `· ·. *   ·   `·*.`·.•·


def parse(s: str) -> list[tuple[str|int, ...]]:
    res = []
    for line in s.split("\n"):
        ins: list[str|int] = []
        for elem in line.strip().split():
            try:
                ins.append(int(elem))
            except ValueError:
                ins.append(elem)
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

    def as_tuple(self):
        return tuple(self[k] for k in self._keys)


class Interpreter:
    def __init__(self, registry: Registry, instructions, verbose=False) -> None:
        """Optimized interpreter which handles multiplication."""

        self.registry = registry
        self.instructions = [ins for ins in instructions]
        self.pointer = 0

        # Stuff for debugging and spotting which instructions are run the most
        self.verbose = verbose
        self.n_its = 0
        self.counts = [0 for _ in self.instructions]
        self.beeps: list[int] = []
        self.seen_states: set[tuple[int, ...]] = set([])

    def _state(self) -> tuple[int, ...]:
        res = (self.pointer,) + self.registry.as_tuple()
        return res

    def _debug(self):
        """Prints details about current instruction and registry values"""
        ins = self.instructions[self.pointer]
        n = self.counts[self.pointer]
        s = f"{self.n_its}: Instruction {self.pointer}: {ins} ({self.registry}). n_calls: {n}."
        print(s)

    def run_single(self):
        """Runs a single instruction"""

        state = self._state()
        if state in self.seen_states:
            raise RuntimeError("Cycle detected")
        self.seen_states.add(state)

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

    def out(self, x):
        """Runs a toggle instruction which modifies the instructions"""
        val = self.registry[x]
        self.beeps.append(val)

        self.pointer += 1

    def mul(self, x, y, z):
        self.registry[z] += self.registry[x]*self.registry[y]
        self.pointer += 1

    def nop(self):
        self.pointer += 1


def check_recurrence(a: int, instructions: list):
    """Checks if the provided value results in a clock signal ([0, 1, ...])"""

    #set up the interpreter
    reg = Registry(a=a)
    int_ = Interpreter(registry=reg, instructions=instructions)

    while True:
        try:
            int_.run_single()
            # If a value is repeated, we'll never get an alternating repeating signal
            signal_alternates = len(int_.beeps) < 2 or int_.beeps[-1] != int_.beeps[-2]
            if not signal_alternates:
                return False
        except RuntimeError:
            # We've already established that the signal alternates, so if it repeats, it's a clock signal
            signal_repeats = len(int_.beeps) > 1 and int_.beeps[0] != int_.beeps[-1]
            return signal_repeats
        #
    #


def brute(instructions: list):
    a = 0

    while True:
        print(a, end="\r")
        cycle = check_recurrence(a=a, instructions=instructions)
        if cycle:
            print(" "*len(str(a)), end="\r")
            return a
        a += 1


def solve(data: str) -> tuple[int|str, None]:
    raw_instructions = parse(data)
    optimized_instructions = peephole_optimize(raw_instructions)

    star1 = brute(instructions=optimized_instructions)
    print(f"The clock signal can be established using a={star1}.")

    star2 = None

    return star1, star2


def main() -> None:
    year, day = 2016, 25
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
