#  ·`·.* ·  ·  `*  ·.•+`·. `*·.   ·  * `· . ·`*  +·`  ·  *  .· + `   · .* · ` ·.
# .*· `.+· .  ·    ··   .  + Opening the Turing Lock * + ·`·  .  + · `•  ·   ·*·
# ` .  · *·•    . ·` · https://adventofcode.com/2015/day/23 *.•··· . •`*+·. ·*  
# ·.*·· `.+`.·   · .•`    ··     ·  •`.·· `·.·+* . • ·*`  ·   ·     +.·   `·   ·


def parse(s: str):
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


def solve(data: str) -> tuple[int|str, int|str]:
    instructions1 = parse(data)
    computer = Computer(instructions=instructions1)
    computer.run_all()
    star1 = computer.state['b']
    print(f"Solution to part 1: {star1}")

    part2state = {'a': 1, 'b': 0}
    computer2 = Computer(instructions=instructions1, state=part2state)
    computer2.run_all()
    star2 = computer2.state['b']
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
