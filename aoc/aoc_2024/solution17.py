# · ` *.·`  .•`   ·        *.`  ·· `.  · · ` +  ·  .· .  ·  ·      . • · ` .·`.·
# ·. ·`    · ·  + .·       `· Chronospatial Computer  *`  · .`+.·    ·  .··*`· .
#  · `·+ .· ·. `  ·  • https://adventofcode.com/2024/day/17   ·`*··  .  ·  `   ·
# .`·  ·   * `  ··  .`*·   ·•  .`   ·     `   · . .`   ·. *·  `· .  ·` + . · ·` 

from __future__ import annotations

from copy import copy
from dataclasses import dataclass


def parse(s: str) -> tuple[State, tuple[int, ...]]:
    regpart, progpart = s.split("\n\n")
    reg: dict[str, int] = {key: int(val) for key, val in (line[9:].split(": ") for line in regpart.splitlines())}
    state = State(A=reg["A"], B=reg["B"], C=reg["C"])
    program = tuple(int(elem) for elem in progpart.split("Program: ")[-1].split(","))
    return state, program


@dataclass(kw_only=True)
class State:
    """Represents a state, including register values, instruction pointer, and values printed"""
    A: int=0
    B: int=0
    C: int=0
    ip: int = 0
    printed: tuple[int, ...] = ()


def resolve_combo_operand(state: State, operand: int) -> int:
    match operand:
        case 7:
            raise RuntimeError
        case 4:
            return state.A
        case 5:
            return state.B
        case 6:
            return state.C
        case _:
            return operand
        #
    #


opcode_map = {
    0: "adv",
    1: "bxl",
    2: "bst",
    3: "jnz",
    4: "bxc",
    5: "out",
    6: "bdv",
    7: "cdv",
}

operations_using_combo_operand = {"adv", "bst", "out", "bdv", "cdv"}


class Emulator:
    def __init__(self, program: tuple[int, ...], verbose=False) -> None:
        self.verbose = verbose
        self.program = program
    
    def vprint(self, *args, **kwargs) -> None:
        if self.verbose:
            print(*args, **kwargs)
    
    def run_program(self, initial_state: State, maxiter=-1) -> State:
        """Given an initial state, runs the entire program.
        Returns the final state after running the program."""
        state = initial_state
        
        n = 0
        while 0 <= state.ip < len(self.program):
            state = self.progress_single_state(state)
            
            n += 1
            if n >= maxiter and maxiter != -1:
                raise RuntimeError
            #
        return state

    def progress_single_state(self, state: State) -> State:
        """Takes a single state and returns the state after executing 1 instruction"""
        state = copy(state)
        # Determine the instruction and operand
        opcode = self.program[state.ip]
        operation = opcode_map[opcode]
        operand = self.program[state.ip + 1]
        
        # Resolve combo operand if appropriate
        use_combo = operation in operations_using_combo_operand
        arg = resolve_combo_operand(state, operand) if use_combo else operand
        
        # Progress instruction pointer (override this when jumping)
        state.ip += 2
        
        # Execute the instruction for the current opcode
        match operation:
            case "adv":
                state.A = state.A >> arg
            case "bxl":
                state.B = state.B ^ arg
            case "bst":
                state.B = arg % 8
            case "jnz":
                # Jump here, so set the ip to the argument instead of incrementing it
                if state.A != 0:
                    state.ip = arg
            case "bxc":
                state.B = state.B ^ state.C
            case "out":
                state.printed += (arg % 8,)
            case "bdv":
                state.B = state.A >> arg
            case "cdv":
                state.C = state.A >> arg
            case _:
                raise RuntimeError(f"Invalid opcode: {opcode}")
            #
        
        self.vprint(state)
        return state
    
    def process_loop(self, A=0, B=0, C=0) -> State:
        """Processes a single loop in the program.
        Returns the state at the end of the loop (i.e. when hitting the jnz instruction)"""
        
        # Check the entire program is a loop, and only one character is output per iteration
        jnz_ind = next(i for i in range(0, len(self.program), 2) if opcode_map[self.program[i]] == "jnz")
        assert self.program[-2:] == (3, 0)
        assert sum(self.program[i] == 5 for i in range(0, len(self.program), 2)) == 1
        state = State(A=A, B=B, C=C)
        
        while state.ip != jnz_ind:
            state = self.progress_single_state(state)
        
        return state

    def lowest_quine(self) -> int:
        """Return the lowest quine (program that outputs itself)
        This is computed by running the program in reverse, starting with the condition that
        A must be 0 at the end (or the program would keep printing).
        Processing, at the second to last iteration, the last digit of the program must be output,
        and A end up as the previous value (0).
        In each iteration, 3 bits are popped from A, so every time we move to consider the next digit,
        there are only 7 (2**3 - 1) possible new values for A to consider."""
        
        n_shift = 3
        # Double check that A is only shifted by a single instruction, and shifted by 3 bits
        a_shift_inds = [i for i in range(0, len(self.program), 2) if opcode_map[self.program[i]] == "adv"]
        assert len(a_shift_inds) == 1 and self.program[a_shift_inds[0] + 1] == n_shift
        
        # Start with a zero as the onle valid value (as the program terminates)
        values = [0]
        next_ = []
        final_loop = True
    
        for ind, output in enumerate(reversed(self.program)):
            # A the last loop, we can't have A = 0, or the program would terminate before printing the last digit
            low = 1 if final_loop else 0
            final_loop = False
            
            for value in values:
                # Append all possible combinations of 3 extra bits to last value of A
                offset = value << n_shift
                for extra in range(low, 2**n_shift):
                    x = offset + extra
                    
                    # Check if adding this value would output the correct digit from the program
                    state_post = self.process_loop(A=x)
                    printed = state_post.printed[-1]
                    # Double check that we end up with the same A as before
                    assert state_post.A == value
                    
                    # If we get the correct output, this value continues to the next iteration
                    if printed == output:
                        next_.append(x)
                    #
                #
                
            values = next_
            next_ = []
        
        # There might be multiple quines - return the lowest
        return min(values)
    #



def solve(data: str) -> tuple[int|str, int|str]:
    initial_state, program = parse(data)
    emulator = Emulator(program, verbose=False)
    
    end_state = emulator.run_program(initial_state)
    star1 = ",".join(map(str, end_state.printed))
    
    print(f"Solution to part 1: {star1}")

    star2 = emulator.lowest_quine()
    print(f"Solution to part 2: {star2}")
    
    return star1, star2
    
    
def main() -> None:
    year, day = 2024, 17
    from aocd import get_data
    
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
