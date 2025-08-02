# *⸳ꞏ  `ꞏ⸳.ꞏ         ꞏ `. ꞏ  `+⸳.+ ꞏ ⸳*ꞏ.    ꞏ  . ꞏ ꞏ  •⸳.⸳ •`.ꞏ     ꞏ +  ⸳ꞏ.  ꞏ
# . `  ꞏ.    ⸳ꞏ+ *   `ꞏ `  ⸳  ꞏ Chronal Conversion   ꞏ⸳`. +ꞏ*⸳  . ꞏ  *`ꞏ   • ꞏ⸳ 
# ⸳ꞏ.•⸳+⸳   `. ꞏ ꞏ.⸳•  https://adventofcode.com/2018/day/21 .+ꞏ `⸳.⸳•.    ꞏ. ⸳`.
# ꞏ.⸳    ꞏ+.ꞏ`.⸳ ⸳  ` *.ꞏ   . `ꞏ⸳⸳ꞏ  .ꞏ`+ *ꞏ⸳. +⸳ꞏ.     ꞏ⸳   .`  +*ꞏ  ꞏ.` +`ꞏ.⸳ 

from __future__ import annotations
import functools
import math
import re
from typing import Callable, TypeAlias


functype: TypeAlias = Callable[[list[int], int, int, int], None]
instructiontype: TypeAlias = tuple[str, tuple[int,int,int]]
# For storing machine state (instruction pointer + registry data) in a hashable format
statetype: TypeAlias = tuple[int, tuple[int, ...]]


class LoopExecption(Exception):
    """Exception class for when an infinite loop is detected"""
    pass


def parse(s) -> tuple[int, list[instructiontype]]:
    """Parses the input program."""

    ip_line, *lines = s.splitlines()
    ip_reg = int(ip_line.split("#ip ")[-1])
    
    instructions: list[instructiontype] = []
    for opname, *argstring in (line.split(" ") for line in lines):
        a, b, c = map(int, argstring)
        instruction = (opname, (a, b, c))
        instructions.append(instruction)
    
    return ip_reg, instructions


def peephole(data: str) -> str:
    """Define an optimized version of the machine code"""
    
    # Figure out which register is bound to the instruction pointer
    boundreg = data.splitlines()[0].split("#ip ")[-1]
    
    # Evil regex to spot the evil algo
    parts = (
        r"gtir (?P<div>\d+) (?P<R4>\d) (?P<R3>\d)",
        r"addr (?P=R3) (?P<R1>\d) (?P=R1)",
        r"addi (?P=R1) 1 (?P=R1)",
        r"seti 27 \d+ (?P=R1)",
        r"seti 0 \d+ (?P=R3)",
        r"addi (?P=R3) 1 (?P<R2>\d)",
        r"muli (?P=R2) (?P=div) (?P=R2)",
        r"gtrr (?P=R2) (?P=R4) (?P=R2)",
        r"addr (?P=R2) (?P=R1) (?P=R1)",
        r"addi (?P=R1) 1 (?P=R1)",
        r"seti 25 \d+ (?P=R1)",
        r"addi (?P=R3) 1 (?P=R3)",
        r"seti 17 \d+ (?P=R1)",
        r"setr (?P=R3) \d+ (?P=R4)",
    )
    
    pattern = re.compile("\n".join(parts), flags=re.MULTILINE)
    
    match = pattern.search(data)
    if match is None:
        raise RuntimeError(f"Couldn't locate pattern for optimization")
    
    # Check which values match which free variable in the pattern (might be permutations of my input)
    d = match.groupdict()
    
    pre = data[:match.start()]
    post = data[match.end():]
    
    # Make sure the instruction pointer refers to the correct registry address
    match_criteria = dict(
        R1=boundreg,
    )
    
    for k, v in match_criteria.items():
        if d.pop(k) != v:
            raise RuntimeError(f"Match criterion {k}={v} violated")
        #
    
    # Keep the first lines from the match
    n_lines_keep = 4
    inject_lines = match.group(0).splitlines()[:n_lines_keep]
    
    # Replace the inefficient algorithm with a bit shoft/division operator and an appropriate no. of nops
    inject_lines.append(f"{bsri.__name__} {d['R4']} {d['div']} {d['R4']}")
    n_lines_so_far = (pre.count("\n") + post.count("\n")) + len(inject_lines) - 1
    n_nops_required = data.count("\n") - n_lines_so_far
    for _ in range(n_nops_required):
        inject_lines.append(f"{nop.__name__} 0 0 0")
    
    inject_string = "\n".join(inject_lines)
    res = pre+inject_string+post
    
    return res


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
    
    def __init__(
            self,
            program: list[instructiontype],
            ip_register: int,
            registry: list[int]|None=None,
            ip: int=0):
        """Make an emulator with no associated operations yet
        Much of this is copied over from day 16.
        ip_register is the index of the registry storing the instruction pointer"""
        
        self.program = program
        self.registry = [0 for _ in range(self.n_registers)] if registry is None else registry
        self.ip = ip
        assert len(self.registry) == self.n_registers
        self._ip_register = ip_register
        self.ops: dict[str, functype] = {}
        
        self._states_seen: set[statetype] = set()
        self.valid_values: list[int] = []
        self._values_seen: set[int] = set()

    def register_valid_value(self, val: int):
        """Register that a given value will cause the program to halt"""
        if val in self._values_seen:
            return
        self._values_seen.add(val)
        self.valid_values.append(val)
    
    def register_current_state(self):
        """Compute a state key for the current state. Add to seen states,
        raising an error on infinite loop detection"""
        state_key = (self.ip, tuple(self.registry))
        if state_key in self._states_seen:
            raise LoopExecption
        self._states_seen.add(state_key)
    
    @property
    def state(self) -> statetype:
        return (self.ip, tuple(self.registry))
    
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
        
        # If we hit the comparison that can halt, register the value that'll
        if opname == "eqrr":
            ind, other, _ = ins
            # Just double check we're comparing against register 0
            assert other == 0
            val = self.registry[ind]
            self.register_valid_value(val)

        f = self.ops[opname]
        f(self.registry, *ins)
        
    def run_program(self, n_max: int|None=None, verbose:bool=False, exit_on_loop: bool=True) -> int|None:
        """Executes a program (list of instructions).
        n_max is an optional max number of executions. verbose determines whether to print state
        after each instruction (defaults to False).
        Returns the contents of register 0 after running the program.
        if exit_on_loop, program will exit when an infinite loop is detected, returning None.
            if false, will raise an error upon detecting a loop."""
        
        n_max_ = float("inf") if n_max is None else n_max
        n = 0
        
        while 0 <= self.ip < len(self.program) and  n < n_max_:
            
            # Register current state, checking for loops
            try:
                self.register_current_state()
            except LoopExecption:
                if exit_on_loop:
                    # If we allow normal termination on loops, just return here
                    return None
                else:
                    raise
                #
            
            # Execute the next instruction
            step = self.program[self.ip]
            opname, instruction = step
            msg = f"ip={self.ip} {opname} {' '.join(map(str, instruction))} {self.registry}"
            
            self.registry[self._ip_register] = self.ip
            self.apply_op(opname, instruction)
            # Read out instruction pointer after operation and increment it
            self.ip = self.registry[self._ip_register] + 1
            
            msg += f" -> {self.registry}"
            if verbose:
                print(msg)
                
            n += 1
        
        # If the code halts, return the result (register 0)
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
        extra_funcs = (setr, seti, bsri, nop)
        for f in extra_funcs:
            res.add_operation(f.__name__)(f)
        
        return res


def setr(reg: list[int], a: int, _: int, c: int):
    reg[c] = reg[a]


def seti(reg: list[int], a: int, _: int, c: int):
    reg[c] = a


def nop(_reg: list[int], _a: int, _b: int, _c: int):
    """nop (no operation). This is just to be able to add a line which
    does nothing, to conserve the number of total lines in a program."""
    
    return None


def bsri(_reg: list[int], _a: int, _b: int, _c: int):
    """Bit-shift right."""
    
    frac = _reg[_a] / _b
    val = math.floor(frac)
    _reg[_c] = val
    
    return None


def solve(data: str):
    data = peephole(data)
    
    # We assume the comparison controls exit from program flow, so ensure there's only one
    assert data.count("eqrr") == 1

    ip_register, instructions = parse(data)
    emulator = Emulator.setup(program=instructions, ip_register=ip_register)
    
    emulator.run_program(verbose=False, exit_on_loop=True)
    
    star1 = emulator.valid_values[0]
    print(f"Solution to part 1: {star1}")
    
    star2 = emulator.valid_values[-1]
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2018, 21
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
