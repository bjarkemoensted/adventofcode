from collections import defaultdict


def parse(s):
    """Parses input to a list of instructions like [(operation, arg1, arg2, ...)]"""
    res = []
    for line in s.splitlines():
        parts = line.strip().split()
        for i, part in enumerate(parts):
            # Just convert to ints if possible. If that fails, keep as string
            try:
                parts[i] = int(parts[i])
            except ValueError:
                pass
            #
        res.append(tuple(parts))

    return res


class Register(defaultdict):
    """Helper class for registry data, so attempting to look up an int will just return the int"""
    def __missing__(self, key):
        return self.default_factory(key)

    def __str__(self):
        s = "{" + ", ".join([f"{k}: {v}" for k, v in self.items() if v != 0]) + "}"
        return s
    #


def _initialize_register(instructions: list, value: int = 0) -> Register:
    """Creates a register, using instructions to determine the appropriate registers and keys."""

    # Assume we just need a register for each string argument in the instructions
    keys = sorted(set([elem for elem in sum([list(tup[1:]) for tup in instructions], []) if isinstance(elem, str)]))

    reg = Register(lambda x: x)
    for k in keys:
        reg[k] = value

    return reg


def _update_reg(reg: Register, instruction: tuple) -> int:
    """Runs an instruction which modifies the registry."""

    # Parse args
    op, *args = instruction
    if len(args) == 1:
        x = args[0]
    elif len(args) == 2:
        x, y = args

    # Default to skipping one instruction ahead after running
    inc = 1

    if op == "set":
        reg[x] = reg[y]
    elif op == "sub":
        reg[x] -= reg[y]
    elif op == "mul":
        reg[x] *= reg[y]
    elif op == "jnz":
        # Conditional jump
        if reg[x] != 0:
            inc = reg[y]
        #
    elif op == "uglyop":
        b, d, e, f, g = args
        reg[g] = 0
        reg[e] = reg[b]
        if reg[b] % reg[d]:
            reg[f] = 0
        #
    elif op == "nop":
        pass
    else:
        raise ValueError(f"Couldn't run operation {op} with args {args}")

    return inc


def run_instructions(instructions: list, verbose=False, max_ins=None, **regvals):
    """Runs the instructions"""

    reg = _initialize_register(instructions)
    for k, v in regvals.items():
        reg[k] = v

    if max_ins is None:
        max_ins = float("inf")

    n_muls = 0
    n_run = 0
    linecounts = defaultdict(lambda: 0)

    ind = 0

    while 0 <= ind < len(instructions):
        linecounts[ind] += 1
        op, *_ = instructions[ind]
        n_muls += op == "mul"

        inc = _update_reg(reg, instructions[ind])
        ind += inc
        n_run += 1
        if n_run >= max_ins:
            break
        #

    # Display the instructions and number of executions (to help identify where to optimize)
    if verbose:
        print()
        for ind, ins in enumerate(instructions):
            print(f"{ind}: ({linecounts[ind]}) - {instructions[ind]}")
        print()

    return n_muls


# General pattern for the instructions, to match variables etc
sum_nonprimes_pattern = [
    ('set', 'b', 'val_b0'),
    ('set', 'c', 'b'),
    ('jnz', 'a', 2),
    ('jnz', 1, 5),
    ('mul', 'b', 'val_b_factor'),
    ('sub', 'b', 'val_b_offset'),
    ('set', 'c', 'b'),
    ('sub', 'c', 'val_c_offset'),
    ('set', 'f', 1),
    ('set', 'd', 2),
    ('set', 'e', 2),
    ('set', 'g', 'd'),
    ('mul', 'g', 'e'),
    ('sub', 'g', 'b'),
    ('jnz', 'g', 2),
    ('set', 'f', 0),
    ('sub', 'e', -1),
    ('set', 'g', 'e'),
    ('sub', 'g', 'b'),
    ('jnz', 'g', -8),
    ('sub', 'd', -1),
    ('set', 'g', 'd'),
    ('sub', 'g', 'b'),
    ('jnz', 'g', -13),
    ('jnz', 'f', 2),
    ('sub', 'h', -1),
    ('set', 'g', 'b'),
    ('sub', 'g', 'c'),
    ('jnz', 'g', 2),
    ('jnz', 1, 3),
    ('sub', 'b', 'val_b_inc'),
    ('jnz', 1, -23),
]


def match(instructions, ind=0, seq=None, mapping=None):
    """The instructions count the number of non-primes bewteen 2 numbers (but do so very slowly).
    This method attempts to recognize a generalization of the instructions from my input.
    This works by comparing the pattern to the input instructions and noting whether the pattern
    matches. This is done by comparing each operation, and maintaining a mapping from variables in the pattern
    to variables in the actual instructions. Similarly, placeholders, such as 'val_b0' are used to indicate integers
    in the input instructions. If contradictions arise (such as 'g' mapping to several other characters), no match
    is possible."""

    # Start with the full sequence in the pattern and an empty mapping
    if seq is None:
        seq = sum_nonprimes_pattern
    if mapping is None:
        mapping = dict()

    # If there's nothing left to match, the pattern works
    if len(seq) == 0:
        return mapping

    # Grab the operation and any arguments from the pattern and instruction
    op, *args = instructions[ind]
    target = seq[0]
    op_target, *args_target = target

    # No match if the operations or number of args differ
    if op != op_target or len(args) != len(args_target):
        return None

    # Compare the arguments
    for x, y in zip(args, args_target):
        # Integers must match exactly
        if all(isinstance(par, int) for par in (x, y)):
            if isinstance(x, int):
                if x != y:
                    return None
                continue
            #
        # If the pattern has an integer and the instruction a register, no general match is possible
        if isinstance(x, str) and isinstance(y, int):
            return None

        # If this part of the pattern has already been mapped to a different value, no match is possible
        if y in mapping and mapping[y] != x:
            return None

        # Otherwise, note the mapping and continue
        mapping[y] = x

    # Continue with the remaining n-1 elemens of the pattern and instructions
    return match(instructions, ind+1, seq[1:], mapping)


def match_template_and_run(instructions: list) -> int:
    """Matches the pattern instructions, and performs the equivalent computation by summing the number of non-primes
    bewtween the values set at the beginning of the instructions."""

    pars = match(instructions)

    b = pars["val_b0"]*pars["val_b_factor"] - pars["val_b_offset"]
    c = b - pars["val_c_offset"]
    b_inc = -pars["val_b_inc"]

    def is_prime(n):
        return n > 1 and all(n % div != 0 for div in range(2, round(n ** 0.5) + 1))

    res = sum(not is_prime(n) for n in range(b, c+1, b_inc))

    return res


def solve(data: str):
    instructions = parse(data)

    star1 = run_instructions(instructions)
    print(f"Solution to part 1: {star1}")

    star2 = match_template_and_run(instructions)
    # 914 too low
    # TODO count n non-primes between b and c (including c I think)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
