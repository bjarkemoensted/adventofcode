_test = """cpy 2 a
tgl a
tgl a
tgl a
cpy 1 a
dec a
dec a"""


def read_input():
    #return _test #!!!
    with open("input23.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def _toggle(instruction):
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


def isvalid(instruction):
    fun, *args = instruction
    if fun == "cpy":
        return isinstance(args[1], str)
    elif fun in ("inc", "dec"):
        return isinstance(args[0], str)
    return True


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


class Interpreter:
    def __init__(self, registers: dict, instructions, pointer=0, verbose=False):
        self.registers = {k: v for k, v in registers.items()}
        self.keys = sorted(self.registers.keys())
        self.instructions = [ins for ins in instructions]
        self.pointer = pointer
        self._seen_states = set([])
        self.verbose = verbose
        self.n_its = 0

    def _debug(self):
        reg = ", ".join(f"{k}={self.registers[k]}" for k in self.keys)
        s = f"Step: {self.n_its}: {reg}. Ind {self.pointer}. Current instruction {self.instructions[self.pointer]}."
        print(s)

    @property
    def contents(self):
        res = tuple(self.registers[k] for k in self.keys)
        return res

    def _tup(self):
        res = (self.pointer, tuple(self.instructions), self.contents)
        return res

    def run_single(self):
        instruction = self.instructions[self.pointer]
        state = self._tup()
        if self.verbose:
            self._debug()

        if state in self._seen_states:
            raise RuntimeError("Circular instructions detected")
        self._seen_states.add(state)

        if not isvalid(instruction):
            raise ValueError(f"Invalid instruction: {instruction}")

        ins, *args = instruction
        fun = getattr(self, ins)
        fun(*args)
        self.n_its += 1

    def run_instructions(self):
        done = False
        while not done:
            self.run_single()
            done = not (0 <= self.pointer < len(self.instructions))

    def _read(self, x):
        val = x
        if isinstance(x, str):
            val = self.registers[x]
        return val

    def tgl(self, x):
        shift = self._read(x)
        target_ind = self.pointer + shift
        if 0 <= target_ind < len(self.instructions):
            target_ins = self.instructions[target_ind]
            new_ins = _toggle(target_ins)
            self.instructions[target_ind] = new_ins

        self.pointer += 1

    def cpy(self, x, y):
        val = self._read(x)
        self.registers[y] = val
        self.pointer += 1

    def inc(self, x):
        self.registers[x] += 1
        self.pointer += 1

    def dec(self, x):
        self.registers[x] -= 1
        self.pointer += 1

    def jnz(self, x, y):
        x = self._read(x)
        y = self._read(y)
        if x == 0:
            self.pointer += 1
        else:
            self.pointer += y


def main():
    raw = read_input()
    instructions = parse(raw)
    reg = {let: 0 for let in "abcd"}
    reg["a"] = 7

    int_ = Interpreter(registers=reg, instructions=instructions, verbose=True)
    int_.run_instructions()
    star1 = int_.registers["a"]
    print(f"After running the instructions, register a contains the value {star1}.")

    #reg["a"] = 12
    #int2 = Interpreter(registers=reg, instructions=instructions, verbose=True)
    #int2.run_instructions()
    #star2 = int2.registers["a"]
    #print(f"After running the instructions, register a contains the value {star2}.")


if __name__ == '__main__':
    main()
