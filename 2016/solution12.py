test = """cpy 41 a
inc a
inc a
dec a
jnz a 2
dec a"""


def read_input():
    with open("input12.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


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
    def __init__(self, registers: dict, instructions, pointer=0):
        self.registers = {k: v for k, v in registers.items()}
        self.instructions = instructions
        self.pointer = pointer

    def run_instructions(self):
        done = False
        while not done:
            ins, *args = self.instructions[self.pointer]
            fun = getattr(self, ins)
            fun(*args)
            done = not (0 <= self.pointer < len(self.instructions))

    def _read(self, x):
        val = x
        if isinstance(x, str):
            val = self.registers[x]
        return val

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
    int_ = Interpreter(registers=reg, instructions=instructions)

    int_.run_instructions()
    star1 = int_.registers["a"]
    print(f"After running the instructions, register a contains the value {star1}.")

    reg["c"] = 1
    int2_ = Interpreter(reg, instructions)
    int2_.run_instructions()
    star2 = int2_.registers["a"]
    print(f"With the new initialization, register a ends up containing {star2}.")


if __name__ == '__main__':
    main()
