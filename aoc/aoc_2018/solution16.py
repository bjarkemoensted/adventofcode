# ·.+  · `·  `.* ·`   .  · +`  *. ·. ·`  `     .·+.  `• · .  ·*` •.    ·.· `·+*.
# .+·  .·*   ·`   +·`. · .  · Chronal Classification  `·.  `*.   ··   *` .· `.··
# .· · ``·+·    ·.·    https://adventofcode.com/2018/day/16 · . ·   •`·+  .· `.·
# `·*   ·. `· ·. ` *  +`.  ·  +· `.+  · `· • .· ` ·   ·    .` ·•  + ·.    `·*·`.

import functools
from itertools import groupby
from typing import Callable, TypeAlias


functype: TypeAlias = Callable[[list[int], int, int, int], None]


def parse(s: str):
    """Input is in 2 parts:
        a: A section with instructions and register values before/after applying the instruction,
        b: A section with a number of instructions"""
    
    a, b = s.split("\n\n\n")
    # parse the last program first
    parts = a.split("\n\n")
    program = [list(map(int, line.split())) for line in b.strip().splitlines()]

    # Parse the tuples of register vals and instructions from the first part
    samples = []
    for part in parts:
        entries = [line.split(": ")[-1].replace("[", "").replace("]", "").replace(",", "") for line in part.splitlines()]
        regs = tuple(list(map(int, ent.split())) for ent in entries)
        if not regs:
            continue
        samples.append(regs)
    
    for meh in samples:
        if len(meh) != 3:
            print(meh)
            raise RuntimeError

    return samples, program


class Emulator:
    """Emulates the various opcodes provided in the problem"""
    
    def __init__(self):
        """Make an emulator with no associated operations yet"""
        self.opnames = []
        self.ops = {}
    
    def _add_op(self, name: str, func: functype):
        """Register an operation under the provided opname"""

        if name in self.ops:
            raise RuntimeError(f"Operation '{name}' is already registered")
        
        self.opnames.append(name)
        self.ops[name] = func

    def add_operation(self, name: str):
        """Adds a function to the emulator's ops. This can be used as a decorator via
        @add_operation("my_opname")
        def _(...):
            # code here
        """
        
        # Given the name specified (as the decorator parameter), create a decorator which
        # registers the wrapped function, and return that
        def decorator(f):
            @functools.wraps(f)
            def wrapped(reg: list[int], a: int, b: int, c: int) -> None:
                f(reg, a, b, c)
                return None
        
            self._add_op(name=name, func=wrapped)
            return wrapped
        return decorator

    def add_op_with_magic_method(self, method: str, name: str, variant: str):
        """Register an operation based on a built in 'magic method' (e.g. '__add__' for addition, etc).
        This works be defining a new function which applies the provided magic method to a register,
        then registering it to the emulator under the provided name.
        'variant' can be one or two characters 'r'/'i', denoting whether the operation whould work
        with 'register' or 'immediate' values, i.e. if an argument is 2 whether the value used should
        be the integer 2, or the value in register 2, respectively."""

        # Make a name for the function (e.g. 'add' with variant 'r' -> 'addr')
        full_name = name+variant
        # If the variant only has 1 char, it's assumed the symbol refers to the b arg, and the a arg is 'r'
        spec = variant if len(variant) == 2 else "r"+variant
            
        def op(reg: list[int], a: int, b: int, c: int) -> None:
            # Use the specification string thingy to grab registry or literal values based on a/b
            aval, bval = (reg[x] if s == "r" else x for x, s in zip((a, b), spec, strict=True))

            # Apply the magic method and store the result under c
            res = int(aval.__getattribute__(method)(bval))
            reg[c] = res
        
        # Register the created function to the emulator
        self._add_op(name=full_name, func=op)

    def apply_op(self, opname: str, *args, **kwargs):
        """Applies the operation registerd under the provided name."""
        
        f = self.ops[opname]
        res = f(*args, **kwargs)
        return res
    
    def run_program(self, mapping: dict[int, str], program: list[list[int]]):
        """Executes a program (list of instructions), using the provided mapping between
        opcodes and operation names."""
        reg = [0, 0, 0, 0]
        for opname, a, b, c in program:
            op = mapping[opname]
            self.apply_op(op, reg, a, b, c)
        
        return reg
    

# Make an emulator instance for doing the computations
emulator = Emulator()

# The builtin operators based on which most operations are made
logic = [
    ("__add__", "add"),
    ("__mul__", "mul"),
    ("__and__", "ban"),
    ("__or__", "bor"),
    ("__gt__", "gt"),
    ("__eq__", "eq"),
]

# For the two comparison operators ('=', ">"), both args a and b can refer to register index or literal value
comparisons = ("gt", "eq")

# Construct most of the operations and register them to the emulator
for meth, op in logic:
    variants = ("ir", "ri", "rr") if op in comparisons else ("r", "i")
    for variant in variants:
        emulator.add_op_with_magic_method(method=meth, name=op, variant=variant)
    #


# Couldn't figure out a simple way of defining the setter ops in a similar way, so adding these manually
@emulator.add_operation("setr")
def _(reg: list[int], a: int, b: int, c: int):
    reg[c] = reg[a]


@emulator.add_operation("seti")
def _(reg: list[int], a: int, b: int, c: int):
    reg[c] = a


def determine_compatible_ops(samples: list[tuple]) -> list[tuple[int, set[str]]]:
    """For each data point in the sample - tuple of (register before, instruction, register after),
    determine which operations could cause the observed change in registry values.
    Returns a list of (opcode, set_of_compatible_ops)."""

    res = []
    for sample in samples:
        # Extract register pre/post values and instruction from each sample
        before, instruction, after = sample
        compatible = set()
        for opname in emulator.opnames:
            # Run each operation on the pre-registry (use a copy, as ops manipulate in-place)
            reg = [val for val in before]
            opcode, a, b, c = instruction
            emulator.apply_op(opname, reg, a, b, c)

            # If the computed value matches the post-op register values from the sample, it's a match
            if reg == after:
                compatible.add(opname)
        
        res.append((opcode, compatible))
    
    return res


def link_opcodes_to_ops(compatible: list[tuple[int, set[str]]]) -> dict[int,str]:
    """Determine which opcodes correspond to which operations, using the sample results."""
    
    # For each opcode encountered, generate a set of ops that match all the transformation observed for that opcode
    matches = {k: set.intersection(*(s for _, s in v)) for k, v in groupby(sorted(compatible), key=lambda t: t[0])}

    # Use method of exclusion to repeatedly associate opcodes with only one caompatible op
    res: dict[int,str] = dict()

    while matches:
        # Pop the next opcode with only one possible operation
        opcode = next(k for k, v in matches.items() if len(v) == 1)
        ops = matches.pop(opcode)
        op = list(ops)[0]
        assert op not in res.values()
        
        # Add the opcode to result, and remove it from possible ops for remaining data
        res[opcode] = op
        for k, v in matches.items():
            matches[k] = v - ops
        #    

    return res


def solve(data: str) -> tuple[int|str, int|str]:
    samples, program = parse(data)
    compatible = determine_compatible_ops(samples)

    star1 = sum(len(comp_ops) >= 3 for _, comp_ops in compatible)
    print(f"Solution to part 1: {star1}")

    # Map opcodes to operations and run the instructions
    mapping = link_opcodes_to_ops(compatible)
    output = emulator.run_program(mapping, program)
    
    # The answer is the value of register 0
    star2 = output[0]
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 16
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
