import inspect
import operator
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Operation:
    """Operation for the Intcode computer. This is just to
    have a uniform interface for getting number of parameters etc"""
    
    func: Callable
    n_params: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_params = len(inspect.signature(self.func).parameters)

    def __call__(self, *args: int) -> int:
        res = self.func(*args)
        return res


opcodes = {
    1: Operation(operator.add),
    2: Operation(operator.mul)
}


class Computer:
    STEPSIZE = 4
    STOP = 99

    def __init__(self) -> None:
        self.memory: list[int] = []

    def initialize_memory(self, program: list[int]) -> None:
        self.memory = [val for val in program]

    def run_instruction(self, ptr: int) -> int:
        """Runs a single instruction"""
        opcode = self.memory[ptr]
        if opcode == self.STOP:
            return -1
        
        operation = opcodes[opcode]

        # TODO adapt to variable number of parameters when necessary
        stopat = ptr + 1 + operation.n_params + 1
        _, address_a, address_b, target_address = self.memory[ptr:stopat]
        a = self.memory[address_a]
        b = self.memory[address_b]

        result = operation(a, b)

        self.memory[target_address] = result
        return ptr + self.STEPSIZE

    def run(self, program: list[int]) -> int:
        """Runs an Intcode program"""
        self.initialize_memory(program)
        instruction_pointer = 0
        while 0 <= instruction_pointer < len(self.memory):
            instruction_pointer = self.run_instruction(instruction_pointer)

        res = self.memory[0]
        return res
    #
