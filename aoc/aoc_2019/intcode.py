import inspect
from collections import deque
from enum import IntEnum
from functools import cache
from typing import Callable, Iterable, NamedTuple, Self


class Mode(IntEnum):
    """Allowed parameter modes"""
    POSITION = 0
    IMMEDIATE = 1


class Par(NamedTuple):
    """Parameter for the IntCode computer"""
    val: int
    mode: Mode


class Computer:
    def __init__(self, program: Iterable[int], debug=False) -> None:
        self.memory = [elem for elem in program]
        self.ip = 0  # instruction pointer
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

    def __getitem__(self, key: Par):
        """Gets the value of a parameter"""
        if key.mode == Mode.POSITION:
            return self.memory[key.val]
        elif key.mode == Mode.IMMEDIATE:
            return key.val
        else:
            raise ValueError(f"Invalid key: {key}")
        #
    
    def __setitem__(self, key: Par, value: int):
        """Sets a parameter value"""
        if key.mode != Mode.POSITION:
            raise RuntimeError("Attempted to write a parameter in non-position mode")
        self.memory[key.val] = value

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

    def resolve_instruction(self, value: int) -> tuple[Callable, list[Mode]]:
        mod = 100
        opcode = value % mod
        method, n_pars = resolve_opcode(self, opcode)

        temp = value // mod
        modes: list[Mode] = []
        for _ in range(n_pars):
            modes.append(Mode(temp % 10))
            temp //= 10
        
        return method, modes

    def run_instruction(self) -> None:
        """Runs a single instruction"""

        ip = self.ip
        value = self.memory[self.ip]
        self.vprint(f"ip: {self.ip}: {value}")
        method, modes = self.resolve_instruction(value)
        
        pars = tuple(
            Par(val=self.memory[self.ip+i+1], mode=mode)
            for i, mode in enumerate(modes)
        )

        self.vprint(f"ip: {self.ip}, op: {method.__name__}, memory: {self.memory[self.ip:self.ip+4]}, {pars=}")

        method(*pars)
        ip_moved = self.ip != ip
        if not ip_moved:
            self.ip += len(modes) + 1
        
    def run(self) -> Self:
        """Runs an Intcode program"""
        
        while not self.halted:
            self.run_instruction()
        
        return self
     
    def read_memory(self, address: int=0) -> int:
        """Reads from memory, at the specified address"""
        return self.memory[address]
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
        case 99:
            return computer.halt
        case _:
            raise ValueError(f"Invalid opcode: {opcode}")
        #
    #