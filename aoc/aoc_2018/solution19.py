# ·*.·`   + ·`  ` . *·  * · `  · .·`. ·  + ·` +. · * · `   ·`·.  ·   `.•·` * · ·
# .·*• `  ·  .·`·  . *· `· `  ·  Go With The Flow   .`·* ·   .·+   ·  •` * ·.  ·
# *`·. · ` · •.·   · + https://adventofcode.com/2018/day/19 ·  ·`*·.  · * ·`·.•`
# · . ·*·. * ·`  *     ·.* · •`  · .·    ` .  .··  .·*  ` ·     ·`   * ·.·+ ` ·*

from __future__ import annotations
from collections import defaultdict
import functools
import itertools
import re
from typing import Callable, TypeAlias

functype: TypeAlias = Callable[[list[int], int, int, int], None]
instructiontype: TypeAlias = tuple[str, tuple[int,int,int]]


def parse(s, optimize=True) -> tuple[int, list[instructiontype]]:
    """Parses the input program.
    if optimize is True, applies peephole optimization to replace the super slow
    sum-of-divisors algo with a more sneaky one."""

    if optimize:
        s = peephole(s)
    
    ip_line, *lines = s.splitlines()
    ip_reg = int(ip_line.split("#ip ")[-1])
    
    instructions: list[instructiontype] = []
    for opname, *argstring in (line.split(" ") for line in lines):
        a, b, c = map(int, argstring)
        instruction = (opname, (a, b, c))
        instructions.append(instruction)
    
    return ip_reg, instructions


def factor(n: int, factors: dict[int, int]|None=None) -> dict[int, int]:
    """Returns a prime factoring of n, represented as a dict with factors as keys
    and powers as values, so e.g. {2: 3, 5: 1} represents 2^3*5^1 = 40"""
    
    if factors is None:
        factors = defaultdict(int)
    
    # Look for divisors in the range 2..sqrt(n)
    i = 2
    while i*i <= n:
        if n % i == 0:
            # If we find one, divide it out from n, and recurse
            factors[i] += 1
            return factor(n // i, factors)
        i += 1
    else:
        # If we don't find a divisor, n is prime
        factors[n] += 1
    
    return factors


def get_divisors(n: int):
    """Returns sorted list of all divisors of n"""
    prime_facs = factor(n)
    # For each prime factor, generate all multiples that divide n (e.g. from 2^3 generate 2^0, ... 2^3)
    prime_divs = ([fac**i for i in range(order+1)] for fac, order in prime_facs.items())
    # Compute product of all combinations of such divisors
    res = sorted(map(lambda nums: functools.reduce(lambda a, b: a*b, nums), itertools.product(*prime_divs)))
    return res


def setr(reg: list[int], a: int, _: int, c: int):
    reg[c] = reg[a]


def seti(reg: list[int], a: int, _: int, c: int):
    reg[c] = a


def nop(_reg: list[int], _a: int, _b: int, _c: int):
    """nop (no operation). This is just to be able to add a line which
    does nothing, to conserve the number of total lines in a program."""
    
    return None


def sdiv(reg: list[int], a: int, _: int, c: int):
    """Sum divisors. Efficient-ish algo for computing the sum of numbers
    that divide the value at register a, then storing the result in register c."""
    divisors = get_divisors(reg[a])
    reg[c] = sum(divisors)


def peephole(data: str) -> str:
    """Peephole optimizer for the machine codey thing. It checks for patterns of instructions
    which correspond to a highly inefficient algorithm for summing the divisors of an interger.
    Specifically, it looks for machine code corresponding to running a double loop over values
    ranging from 1 to x, summing one part of each pairs of numbers whose product is x.
    This pattern of instructions is replaced by an efficient (ish) algorithm which uses prime
    factorization to generate divisors and compute their sum. To avoid issues with line numbers,
    nops (no-operations) are added to keep the number of lines unchanged."""
    
    # Figure out which register is bound to the instruction pointer
    boundreg = data.splitlines()[0].split("#ip ")[-1]
    
    # Evil regex to spot the evil algo
    parts = (
        r"seti 1 \d+ (?P<R3>\d)",  # start outer loop
        r"seti 1 \d+ (?P<R2>\d)",  # start inner loop
        r"mulr (?P=R3) (?P=R2) (?P<R4>\d)",  # compute R2*R3
        r"eqrr (?P=R4) (?P<R1>\d) (?P=R4)",  # Use result to check if R3 divides R1
        r"addr (?P=R4) (?P<boundreg>\d) (?P=boundreg)",  # If yes, skip next instruction...
        r"addi (?P=boundreg) 1 (?P=boundreg)",  # skip next instruction
        r"addr (?P=R3) (?P<R0>\d) (?P=R0)",  # ...if yes, add R3 to R0
        r"addi (?P=R2) 1 (?P=R2)",  # increment inner loop counter (R2)
        r"gtrr (?P=R2) (?P=R1) (?P=R4)",  # check inner loop condition
        r"addr (?P=boundreg) (?P=R4) (?P=boundreg)",  # if not condition, skip next
        r"seti (?P<inner_loop_start>\d+) \d (?P=boundreg)",  # GOTO inner loop start
        r"addi (?P=R3) 1 (?P=R3)",  # increment outer loop counter (R3)
        r"gtrr (?P=R3) (?P=R1) (?P=R4)",  # check outer loop condition
        r"addr (?P=R4) (?P=boundreg) (?P=boundreg)", # if not condition, skip next
        r"seti (?P<outer_loop_start>\d+) \d+ (?P=boundreg)",  # GOTO outer loop start
        r"mulr (?P=boundreg) (?P=boundreg) (?P=boundreg)"  # breakout
    )
    
    pattern = re.compile("\n".join(parts), flags=re.MULTILINE)
    
    # Identify the sloppy sum-of-divisors algo
    match = pattern.search(data)
    if match is None:
        raise RuntimeError(f"Couldn't locate pattern for optimization")
    
    # Check which values match which free variable in the pattern (might be permutations of my input)
    d = match.groupdict()
    pre = data[:match.start()]
    post = data[match.end():]
    outer_loop_lineno = pre.count("\n") - 1
    
    # Some values must have specific values (GOTOs must point to the correct lines etc.)
    match_criteria = dict(
        boundreg=boundreg,
        outer_loop_start=str(outer_loop_lineno),
        inner_loop_start=str(outer_loop_lineno+1)
    )
    
    # Make sure the pattern lines up correctly with the instructions
    for k, v in match_criteria.items():
        if d.pop(k) != v:
            raise RuntimeError(f"Match criterion {k}={v} violated")
        #
    
    # Inject an optimized version of the algo, with nops to match number of lines
    inject_lines = [f"{sdiv.__name__} {d['R1']} 0 {d['R0']}"]
    n_nops_required = data.count("\n") - (pre.count("\n") + post.count("\n"))
    inject_lines += [f"{nop.__name__} 0 0 0" for _ in range(n_nops_required)]
    inject_string = "\n".join(inject_lines)
    
    # Insert, and just double check the number of lines is unchanged
    optimized = pre+inject_string+post
    assert optimized.count("\n") == data.count("\n")
    
    return optimized


class Emulator:
    """Emulates the various opcodes provided in the problem"""
    
    n_registers = 6
    
    # The builtin operators based on which most operations are made
    _logic = [
        ("__add__", "add"),
        ("__mul__", "mul"),
        ("__and__", "ban"),
        ("__or__", "bor"),
        ("__gt__", "gt"),
        ("__eq__", "eq"),
    ]

    # For the two comparison operators ('=', ">"), both args a and b can refer to register index or literal value
    _comparisons = ("gt", "eq")
    
    def __init__(self, ip_register: int, registry: list[int]|None=None):
        """Make an emulator with no associated operations yet
        Much of this is copied over from day 16.
        ip_register is the index of the registry storing the instruction pointer"""
        
        self.registry = [0 for _ in range(self.n_registers)] if registry is None else registry
        assert len(self.registry) == self.n_registers
        self._ip_register = ip_register
        self.ops: dict[str, functype] = {}
    
    def _add_op(self, name: str, func: functype):
        """Register an operation under the provided opname"""

        if name in self.ops:
            raise RuntimeError(f"Operation '{name}' is already registered")
        
        self.ops[name] = func

    def add_operation(self, name: str):
        """Adds a function to the emulator's ops. This can be used as a decorator via
        @add_operation("my_opname")
        def _(...):
            # code here
        """
        
        # Given the name specified (as the decorator parameter), create a decorator which
        # registers the wrapped function, and return that
        def decorator(f: functype) -> functype:
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

    def apply_op(self, opname: str, ins):
        """Applies the operation registerd under the provided name."""
        
        f = self.ops[opname]
        f(self.registry, *ins)
        
    
    def run_program(self, program: list[instructiontype], n_max: int|None=None, verbose:bool=False) -> int:
        """Executes a program (list of instructions).
        n_max is an optional max number of executions. verbose determines whether to print state
        after each instruction (defaults to False).
        Returns the contents of register 0 after running the program."""
        
        n_max_ = float("inf") if n_max is None else n_max
        n = 0
        ip = 0
        
        while 0 <= ip < len(program) and  n < n_max_:
            # Copy instruction pointer to the register to which it's bound
            self.registry[self._ip_register] = ip
            
            # Execute the next instruction
            step = program[ip]
            opname, instruction = step
            self.apply_op(opname, instruction)
            
            if verbose:
                msg = f"ip={ip} {opname} {' '.join(map(str, instruction))} {self.registry} (n={n})"
                print(msg)
            
            # Read out instruction pointer after operation and increment it
            ip = self.registry[self._ip_register] + 1
            n += 1
        
        res = self.registry[0]
        return res
    
    @classmethod
    def setup(cls, *args, **kwargs) -> Emulator:
        res = cls(*args, **kwargs)
        
        # Construct the operations based on magic methods
        for meth, op in cls._logic:
            variants = ("ir", "ri", "rr") if op in cls._comparisons else ("r", "i")
            for variant in variants:
                res.add_op_with_magic_method(method=meth, name=op, variant=variant)
            #
        
        # Add extra operations (setters, no op, and sum divisors)
        extra_funcs = (setr, seti, nop, sdiv)
        for f in extra_funcs:
            res.add_operation(f.__name__)(f)
        
        return res


def solve(data: str) -> tuple[int|str, int|str]:
    ip_reg, instructions = parse(data)
    emulator = Emulator.setup(ip_reg)
    
    star1 = emulator.run_program(instructions, verbose=True)
    print(f"Solution to part 1: {star1}")

    emulator2 = Emulator.setup(ip_reg, registry=[1, 0, 0, 0, 0, 0])
    star2 = emulator2.run_program(instructions, verbose=False, n_max=None)
    print(f"Solution to part 2: {star2}")
    
    return star1, star2


def main() -> None:
    year, day = 2018, 19
    from aocd import get_data
    raw = get_data(year=year, day=day)
    
    solve(raw)


if __name__ == '__main__':
    main()
