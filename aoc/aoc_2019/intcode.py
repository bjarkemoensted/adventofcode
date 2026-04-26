import inspect
from collections import defaultdict, deque
from enum import IntEnum
from functools import cache
from typing import Callable, Iterable, NamedTuple, Self, overload


class Mode(IntEnum):
    """Allowed parameter modes"""
    POSITION = 0
    IMMEDIATE = 1
    RELATIVE = 2


class Par(NamedTuple):
    """Parameter for the IntCode computer"""
    val: int
    mode: Mode


class Computer:
    def __init__(self, program: Iterable[int], debug=False) -> None:
        self.memory: dict[int, int] = defaultdict(int)
        for i, val in enumerate(program):
            self.memory[i] = val
        self.ip = 0  # instruction pointer
        self.relative_base = 0
        self.stdin: deque[int] = deque()
        self.stdout: deque[int] = deque()
        self.halted = False
        self.debug = debug

    def vprint(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def add_input(self, *args: int) -> Self:
        """Adds values to the input queue. Input operations can then pop elements from the queue"""
        for val in args:
            self.stdin.append(val)
        return self

    def _resolve_address(self, par: Par) -> int:
        """Resolve the memory address of a parameter"""
        match par.mode:
            case Mode.POSITION:
                return par.val
            case Mode.RELATIVE:
                return par.val + self.relative_base
            case _:
                raise RuntimeError(f"Can't resolve address for parameter mode {par.mode}")
            #
        #

    def __getitem__(self, key: Par) -> int:
        """Gets the value of a parameter"""
        match key.mode:
            case Mode.IMMEDIATE:
                return key.val  # Use the raw value when in 'immediate' mode
            case Mode.POSITION | Mode.RELATIVE:
                # In other modes, resolve memory address and return the value there
                adr = self._resolve_address(key)
                return self.memory[adr]
            case _:
                raise ValueError(f"Invalid key: {key}")
            #
        #
    
    def __setitem__(self, key: Par, value: int):
        """Sets a parameter value"""
        # Resolve the address (this raises an error if par is in immediate mode by mistake)
        address = self._resolve_address(key)
        self.memory[address] = value

    def halt(self) -> None:
        self.halted = True

    def add(self, a: Par, b: Par, c: Par) -> None:
        self[c] = self[a] + self[b]

    def mul(self, a: Par, b: Par, c: Par) -> None:
        self[c] = self[a]*self[b]
    
    def input(self, a: Par) -> None:
        val = self.stdin.popleft()
        self[a] = val
    
    def output(self, a: Par) -> None:
        val = self[a]
        self.stdout.append(val)
    
    def jump_if_true(self, a: Par, b: Par) -> None:
        if self[a] != 0:
            self.ip = self[b]
    
    def jump_if_false(self, a: Par, b: Par) -> None:
        if self[a] == 0:
            self.ip = self[b]
    
    def le(self, a: Par, b: Par, c: Par) -> None:
        val = 1 if self[a] < self[b] else 0
        self[c] = val
    
    def eq(self, a: Par, b: Par, c: Par) -> None:
        val = 1 if self[a] == self[b] else 0
        self[c] = val
    
    def adjust_relative_base(self, a: Par) -> None:
        self.relative_base += self[a]

    def resolve_instruction(self, address: int=-1) -> tuple[Callable, list[Par]]:
        """Determine the instruction, and list of parameters including modes, from
        the specified address"""
        
        if address == -1:
            address = self.ip
        value = self.memory[address]
        # Determine opcode (last 2 digits), and n pars
        mod = 100
        opcode = value % mod
        method, n_pars = resolve_opcode(self, opcode)

        # Determine modes for the parameters which will be read by the op
        temp = value // mod
        modes: list[Mode] = []
        pars: list[Par] = []
        for _ in range(n_pars):
            modes.append(Mode(temp % 10))
            mode = Mode(temp % 10)
            address += 1
            par = Par(val=self.memory[address], mode=mode)
            pars.append(par)
            temp //= 10
        
        return method, pars

    @property
    def current_instruction(self) -> Callable:
        ins, _ = self.resolve_instruction()
        return ins

    def __repr__(self) -> str:
        ins = self.current_instruction.__name__
        s = f"IntCode instance at instruction {self.ip} ({self.memory[self.ip]}: {ins})"
        return s

    def run_instruction(self) -> bool:
        """Runs a single instruction. Returns a boolean indicating whether
        the instruction was succesful. False is returned if the computer
        reaches an input instruction, with no data in the input queue."""

        ip = self.ip
        self.vprint(f"ip: {self.ip}: {self.memory[self.ip]}")
        method, pars = self.resolve_instruction()
        if method == self.input and not self.stdin:
            return False
        
        self.vprint(f"ip: {self.ip}, op: {method.__name__}, memory: {self.memory[self.ip]}, {pars=}")

        method(*pars)
        ip_moved = self.ip != ip
        if not ip_moved and not self.halted:
            self.ip += len(pars) + 1
        
        return True
        
    def run(self) -> Self:
        """Runs an Intcode program"""
        
        while not self.halted:
            success = self.run_instruction()
            if not success:
                break
        
        return self
     
    def read_memory(self, address: int=0) -> int:
        """Reads from memory, at the specified address"""
        return self.memory[address]
    
    @overload
    def read_stdout(self, n: None = None) -> int: ...
    @overload
    def read_stdout(self, n: int) -> tuple[int, ...]: ...
    def read_stdout(self, n=None):
        """Reads from the standard output.
        If no n is specified (default), returns a single integer.
        Otherwise, returns a tuple of n elements. n=-1 can be used
        to empty the stdout queue and return everything as a tuple."""

        if n is None:
            return self.stdout.popleft()

        if n == -1:
            n = len(self.stdout)
        return tuple(self.stdout.popleft() for _ in range(n))
    
    #


@cache
def resolve_opcode(computer: Computer, opcode: int) -> tuple[Callable, int]:
    """Take an opcode and return the corresponding method, and the number
    of parameters."""
    method = opcode_lookup(computer, opcode)
    n_pars = len(inspect.signature(method).parameters)
    return method, n_pars


def opcode_lookup(computer: Computer, opcode: int) -> Callable:
    """Resolve a two-digit opcode to the corresponding instruction
    (i.e. a method in the VM class)"""
    match opcode:
        case 1:
            return computer.add
        case 2:
            return computer.mul
        case 3:
            return computer.input
        case 4:
            return computer.output
        case 5:
            return computer.jump_if_true
        case 6:
            return computer.jump_if_false
        case 7:
            return computer.le
        case 8:
            return computer.eq
        case 9:
            return computer.adjust_relative_base
        case 99:
            return computer.halt
        case _:
            raise ValueError(f"Invalid opcode: {opcode}")
        #
    #
