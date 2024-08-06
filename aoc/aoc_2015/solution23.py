import abc
from copy import deepcopy

# Read in data
with open("input23.txt") as f:
    puzzle_input = f.read()


def parse(s):
    res = []
    for line in s.split("\n"):
        words = line.replace(",", "").split(" ")
        inst = [int(w) if w[0] in "+-" else w for w in words]
        res.append(inst)

    return res


def get_default_state():
    d = {"a": 0, "b": 0}
    return d


class Computer:
    def __init__(self, instructions, state=None):
        if not state:
            state = get_default_state()

        self.state = state
        self.instructions = instructions
        self.current_line = 0

    def increment_line(self):
        self.current_line += 1

    def hlf(self, register):
        self.state[register] //= 2
        self.increment_line()

    def tpl(self, register):
        self.state[register] *= 3
        self.increment_line()

    def inc(self, register):
        self.state[register] += 1
        self.increment_line()

    def jmp(self, offset):
        self.current_line += offset

    def jie(self, register, offset):
        register_even = self.state[register] % 2 == 0
        if register_even:
            self.jmp(offset)
        else:
            self.increment_line()

    def jio(self, register, offset):
        register_one = self.state[register] == 1
        if register_one:
            self.jmp(offset)
        else:
            self.increment_line()

    def run_next(self):
        inst, *args = self.instructions[self.current_line]
        fun = getattr(self, inst)
        try:
            fun(*args)
        except Exception:
            print(f"Issue executing instruction {inst} with arguments {args}.")
            raise

    def run_all(self):
        while 0 <= self.current_line < len(self.instructions):
            self.run_next()


instructions1 = parse(puzzle_input)
computer = Computer(instructions=instructions1)
computer.run_all()

print(f"After running instructions, register b has value {computer.state['b']}")

part2state = {'a': 1, 'b': 0}
computer2 = Computer(instructions=instructions1, state=part2state)
computer2.run_all()
print(f"After running instructions starting with a=1, register b has value {computer2.state['b']}")
