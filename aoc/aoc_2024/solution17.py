from collections import deque
import copy
from functools import singledispatchmethod
from itertools import product
import math
import sympy



raw = """Register A: 2024
Register B: 0
Register C: 0

Program: 0,3,5,4,3,0"""


def parse(s):
    #s = raw  # !!!
    print(s)
    regpart, progpart = s.split("\n\n")
    reg = dict()
    for line in regpart.splitlines():
        k, v = line.split("Register ")[1].split(": ")
        reg[k] = int(v)
    
    prog = [int(elem) for elem in progpart.split("Program: ")[1].split(",")]

    return reg, prog


class BoolVar(sympy.Symbol):
    """Represents a boolean variable, with operators for negation, conjunction, etc"""
    
    def __invert__(self):
        return 1 - self
    
    def __and__(self, other):
        return self*other
    
    def __or__(self, other):
        return self + other - self*other

    def __xor__(self, other):
        return self + other - 2*self*other
    #
    
    def bool_constraint(self):
        """Constraint on a bool ('x and not x' must be zero)"""
        constradiction = self*(1 - self)
        const = sympy.Eq(constradiction, 0)
        return const
    
    @classmethod
    def join_with_operator(cls, elems, operator: str):
        """Helper method for iteratively combining elements in an iterable using a given operator.
        For example, calling with [a, b, c], "&", gives a & b & c."""

        attrs = {
            "&": "and",
            "|": "or",
            "^": "xor",
        }
        
        attr = f"__{attrs[operator]}__"
        
        res = None
        for elem in elems:
            if res is None:
                res = elem
            else:
                res = getattr(res, attr)(elem)
            #
        
        return res
    #


def _get_all_bool_value_subs(symbols: list):
    """Takes a list of symbols - generates all dicts mapping each symbol to each possible 0/1 combination"""
    for comb in product((0, 1), repeat=len(symbols)):
        # Insert the corresponding truth values into the sum and evaluate it to obtain a possible integer
        d = dict(zip(symbols, comb))
        yield d
    #


class BitArray:
    """Represents a bit array which can store a combination of 0's, 1's, and binary variables."""
    
    _padval = False
    
    def __init__(self, value: int|str|list=0, n_bits=None, verbose=False):
        """Create a bit array. n_bits is the number of bits.
        value indicates what the bitarray should hold:
            If an integer (default is 0), the bits will be the binary representation of the int.
            If a list, the elements in the list will be stored as bits (padded to match n_bits).
            If string, bits will be binary variables, e.g. "A" -> [..., A_2, A_1, A_0]"""
        
        self._n_bits = n_bits
        self.verbose = verbose
        bits = self.to_bits(value)
        self.bits = deque(bits)
        self._pad()
        assert len(self.bits) == self.n_bits
    
    @property
    def n_bits(self):
        if self._n_bits is None:
            return len(self.bits)
        else:
            return self._n_bits
    
    def _pad(self):
        """Pads the input bit values with 0's until it has the required length."""
        
        n_short = (self.n_bits - len(self.bits))
        
        if n_short >= 0:
            for _ in range(n_short):
                self.bits.appendleft(self._padval)
        else:
            raise RuntimeError(f"Attempted to pad {len(self.bits)} to length {self.n_bits}.")
        #
    
    @property
    def symbolic(self) -> bool:
        """Indicates whether the bit array is symbolic, i.e. contains any sympy expressions in its bit values"""
        return any(isinstance(val, sympy.Expr) for val in self.bits)
    
    @singledispatchmethod
    def to_bits(self, value: None):
        raise RuntimeError
    
    @to_bits.register
    def _(self, value: int):
        """Converts an integer to a list of bits"""
        bits = [bool(int(char)) for char in format(value, 'b')]
        return bits

    @to_bits.register
    def _(self, value: list):
        bits = [v for v in value]
        return bits
    
    @to_bits.register
    def _(self, value: str):
        bits = [sympy.Symbol(f'{value}_{i}', binary=True) for i in reversed(range(self.n_bits))]
        return bits
    
    def value(self):
        """Computes the sum b_i * 2^i where b_i is the value of the i'th bit from the right.
        Gives an integer if the bitarray is not symbolic, otherwise a sympy expression."""
        return sum(bool(bit)*2**i for i, bit in enumerate(reversed(self.bits)))
    
    def copy(self):
        res = copy.deepcopy(self)
        return res
    
    def __repr__(self):
        return ' '.join(map(str, self.bits))

    def __str__(self):
        return f"Bit array: {repr(self)}"

    @property
    def free_symbols(self):
        syms = set()
        for bit in self.bits:
            try:
                syms |= bit.free_symbols
            except AttributeError:
                pass
        
        res = sorted(syms, key=str)
        return res

    def possible_values(self):
        """Iterates over pairs of possible_value, assumption for a symbolic array.
        This is done by Inserting every combination of 0 and 1 on all free variables in the bits.
        Each possible value is an integer which may be stored in the array.
        Each assumption the corresponding values of the boolean variables, expressed in Boolean algebra."""

        # Figure out which free variables there are in the integer this array represents
        assert self.symbolic
        symbols = self.free_symbols
        
        # Array with all (x, not x) for all free variables x
        symbols_and_negations = [[~sym, sym] for sym in symbols]
        
        # Grab all combinations
        for comb in product((0, 1), repeat=len(symbols)):
            # Combine all the truth values into a single expression (by using conjunction)
            assumptions = [symbols_and_negations[i][val] for i, val in enumerate(comb)]
            single_assumption = BoolVar.join_with_operator(assumptions, "&")
            
            # Insert the corresponding truth values into the sum and evaluate it to obtain a possible integer
            d = dict(zip(symbols, comb))
            val = 0
            for i, bit in enumerate(reversed(self.bits)):
                try:
                    base_ = bool(bit.subs(d))
                except AttributeError:
                    base_ = bit
                val += base_*2**i
            yield val, single_assumption
        #
        
    def subs(self, values: dict):
        """Takes a dict mapping boolean variables to values.
        Returns a bit array which results from replacing the variables with their values."""
        
        symbols_insert = set(values.keys())
        bits = []
        for bit in self.bits:
            # Replace values in all bits that aren't an integer
            try:
                overlap = len(bit.free_symbols & symbols_insert) > 0
                if overlap:
                    b = bool(bit.subs(values))
                else:
                    b = bit
            except AttributeError:
                b = bit
            bits.append(b)
        
        # Create a new bin array with the bits obtained
        res = BitArray(value=bits, n_bits=self.n_bits)

        return res
    
    def make_equal_to_condition(self, other):
        """Sets the bitarray equal to another quantity (bitarray or int).
        Returns the logical condition that the two be equal, i.e. the conjunction
        or equality of all bits."""
        
        if isinstance(other, int):
            n_bits_needed = max(self.n_bits, len(format(other, 'b')))  # !!!
            if self.verbose:
                print(f"!!! REQUIRING {other} == {self.bits}")
            other = BitArray(n_bits=n_bits_needed, value=other)
        
        conditions = []
        padded = copy.deepcopy(self.bits)
        while len(padded) < n_bits_needed:
            padded.appendleft(self._padval)
        
        
        # Require equality for each bit
        for lhs, rhs in zip(padded, other.bits, strict=True):
            c = lhs
            if rhs is True:
                pass
            else:
                if isinstance(lhs, bool):
                    c = not lhs
                else:
                    c = ~lhs
                #
            if c is not True:
                # Don't add conditions that are tautologically true
                conditions.append(c)
        
        cond = BoolVar.join_with_operator(elems=conditions, operator="&")
        return cond
    
    def solve_equal(self, other):
        """Sets this bit array equal to another bit array or int (ints are cast as bit arrays automatically).
        Attempts to find a combinations of binary values which can be assigned to all free variables such that
        the equality holds."""

        cond = self.make_equal_to_condition(other=other)
        solution = sympy.satisfiable(cond, all_models=True)
        return list(solution)
    
    def _shift_bits(self, n: int):
        """Shifts bits right by an integer amount (negative n for left shift)."""

        right = n >= 0
        res = self.copy()
        
        pop = res.bits.pop if right else res.bits.popleft
        push = res.bits.appendleft if right else res.bits.append
        for _ in range(abs(n)):
            pop()
            push(self._padval)
        
        return res
    
    def __lshift__(self, other):
        """Shifts n bits to the left"""
        return self.__rshift__(other=other, opposite=True)
    
    def __rshift__(self, other, opposite=False):
        """Shifts n bits to the right"""
        
        # Handle cases where we're shifting by a known integer number of bits
        if isinstance(other, BitArray) and not other.symbolic:
            other = other.value()
        if isinstance(other, int):
            other = -other if opposite else other
            return self._shift_bits(n=other)
        
        # Handle cases where we shift by a bit array with unknowns
        conditionals = [[] for _ in self.bits]
        
        # Identify each possible value the other array can represent, and the corresponding assumption on bit values
        for val, assumption in other.possible_values():
            # Shift by this number of bits
            n = -val if opposite else val
            shifted = self._shift_bits(n)
            # For each bit, append the conditional value of the bit, and the assumptions on which it is conditional
            for i, bit in enumerate(shifted.bits):
                conditionals[i].append(assumption&bit)
            #
        
        # For each bit, combine each possible value*assumption part with the or-operator, to represent all possible values
        bits = [BoolVar.join_with_operator(bit_possibilities, "|") for bit_possibilities in conditionals]
        
        # Convert into a new bit array and return it
        res = BitArray(n_bits=self.n_bits, value=bits)
        
        return res
    
    @property
    def significant_bits(self):
        for bit, n in zip(self.bits, range(len(self.bits), 0, -1)):
            significant = True
            try:
                significant = bool(bit)
            except:
                pass
            if significant:
                return n
        
        
    def __mod__(self, other: int):
        """Takes the modulo with an integer. Assumes (for now) that the integer is a power of 2."""
        if not isinstance(other, int):
            raise TypeError
        
        pow = round(math.log2(other))
        power_of_two = 2**pow == other
        
        if power_of_two:
            res = self.copy()
            # Take the bits to the right of the non-zero bit (e.g. modulo 8 keeps the 3 rightmost bits)
            keep_bits = list(self.bits)[-pow:]
            
            res = BitArray(value=keep_bits)
        else:
            raise NotImplementedError

        return res
    
    def __xor__(self, other):
        if isinstance(other, int):
            other = BitArray(n_bits=self.n_bits, value=other)
        
        a, b = (copy.deepcopy(ba.bits) for ba in (self, other))
        n = max(map(len, (a, b)))
        for ba in (a, b):
            while len(ba) < n:
                ba.appendleft(self._padval)
        
        res_bits = [x^y for x, y in zip(a, b, strict=True)]
        res = BitArray(value=res_bits)
        return res


class Computer:
    registers = ("A", "B", "C")
    combo_operands = (0, 1, 2, 3, "A", "B", "C", None)
    opcodes = ("adv", "bxl", "bst", "jnz", "bxc", "out", "bdv", "cdv")
    ops_with_combo_operands = ("adv", "bdv", "cdv", "bst", "out")
    
    def __init__(self, register_data: dict=None, verbose=False):
        self._reg = {k: 0 for k in self.registers}
        if register_data:
            self.set_register_values(**register_data)
        self.verbose = verbose

    def set_register_values(self, **kwargs):
        for k, v in kwargs.items():
            self._reg[k] = v
        #
    
    def get_register_values(self):
        res = {k: self._reg[k] for k in self.registers}
        return res

    def __setitem__(self, key, val):
        if key not in self._reg:
            raise KeyError
        self._reg[key] = val
    
    def __getitem__(self, key):
        return self._reg[key] if isinstance(key, str) else key

    def adv(self, co):
        self["A"] = self["A"] >> self[co]
    
    def bdv(self, co):
        self["B"] = self["A"] >> self[co]
    
    def cdv(self, co):
        self["C"] = self["A"] >> self[co]
    
    def bxl(self, lo):
        self["B"] = self["B"] ^ lo
    
    def bst(self, co):
        self["B"] = self[co] % 8
    
    def jnz(self, lo):
        return None if self["A"] == 0 else lo
    
    def bxc(self, _):
        self["B"] = self["B"] ^ self["C"]
    
    def out(self, co):
        res = self[co] % 8
        return res

    def __str__(self):
        s = f"Computer with register vals: {self._rep_regs()}"
        return s

    def preprocess_instructions(self, program):
        """Parses instructions into a dict mapping each valid instruction pointer value
        to a tuple of instruction, operand,
        where the operand has been resolved (into a string if it refers to a register value)."""
        
        res = dict()
        for ind in range(0, len(program), 2):
            opcode, operand = program[ind], program[ind + 1]
            instruction = self.opcodes[opcode]
            op = self.combo_operands[operand] if instruction in self.ops_with_combo_operands else operand
            res[ind] = (instruction, op)
        return res

    def iterate_instructions(self, instructions: dict, instruction_pointer=None):
        if instruction_pointer is None:
            instruction_pointer = 0
            
        inds = sorted(instructions.keys())
        nextind = {ind: inds[i+1] if i+1 < len(inds) else None for i, ind in enumerate(inds)}
        while instruction_pointer is not None:
            
            try:
                instruction, operand = instructions[instruction_pointer]
            except KeyError:
                return
            
            fun = getattr(self, instruction)
            res = fun(operand)
            yield instruction_pointer, instruction, res
            
            jump = (instruction == "jnz" and res is not None)
            instruction_pointer = res if jump else nextind[instruction_pointer]
        #

    def run_instructions(self, program: list):
        instructions = self.preprocess_instructions(program=program)
        output = []
        
        for _, instruction, res in self.iterate_instructions(instructions=instructions):
            if instruction == "out":
                output.append(res)
        
        return output
    #


def run(register_data, program):
    com = Computer(register_data=register_data)
    output = com.run_instructions(program)
    res = ','.join(map(str, output))
    return res


def find_quine(program, remaining=None, registers=None, ins_ptr=None):
    
    if remaining is None:
        remaining = [elem for elem in program]
    else:
        remaining = [elem for elem in remaining]
    
    print(f"REMAINING: {len(remaining)}")
    
    if registers is None:
        registers = {k: 0 for k in Computer.registers}
        registers["A"] = BitArray(value="A", n_bits=100)
    else:
        registers = copy.deepcopy(registers)
    
    com = Computer(register_data=registers)
    
    ins = com.preprocess_instructions(program)
    
    # Figure out which register is printed
    out_reg = [opc for ins_, opc in ins.values() if ins_ == "out"]
    assert len(out_reg) == 1
    out_reg = out_reg[0]
    
    instruction_gen = com.iterate_instructions(instructions=ins, instruction_pointer=ins_ptr)
    for ptr, ins_, res in instruction_gen:
        
        if ins_ == "out":
            target = remaining.pop(0)
            condition = com[out_reg].make_equal_to_condition(target)
            solutions = sympy.satisfiable(condition, all_models=True)

            next_ptr = ptr + 2
            for sol in solutions:
                if not sol or sol == {None: True}:
                    continue
                print(BitArray(value="A", n_bits=100).subs(sol).value())
                current_reg = {k: v.subs(sol) for k, v in com.get_register_values().items()}
                yield from find_quine(program=program, remaining=remaining, registers=current_reg, ins_ptr=next_ptr)

        if ins_ == "jnz":
            if not remaining:
                terminate = com["A"].make_equal_to_condition(0)
                break
            #
        #
    #


def lowest_quine(program):
    print(Computer().preprocess_instructions(program))
    best = float("inf")
    for quine in find_quine(program=program):
        pass
    
    
    return best
    
    


def solve(data: str):
    register_data, program = parse(data)

    star1 = run(register_data=register_data, program=program)
    # assert star1 == '2,1,3,0,5,2,3,7,1'  # !!!
    print(f"Solution to part 1: {star1}")

    
    star2 = lowest_quine(program=program)
    
    print(f"Solution to part 2: {star2}")

    return star1, star2
    
def main():
    year, day = 2024, 17
    #from aoc.utils.data import check_examples
    #check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
