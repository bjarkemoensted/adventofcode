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
        #
    
    #


def solve(data: str):
    instructions = parse(data)
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

    return star1, star2


def main():
    year, day = 2016, 12
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
