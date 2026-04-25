# ·`*·`.+ * ` ·   · •` · *`    ·`. + ·`  · .*   `+·    . `·• `·    *. +· ` ·* *·
# `· * `.·  *      .  *•·  *` Amplification Circuit   ··   ` · . *`    .`·*. ·`·
# . .  *·  `·   •  *.· https://adventofcode.com/2019/day/7     ·*.•·  ` . +  `· 
# ·.` *·`   •· +.`  *  ·`  ·  * •· .` * ·.`*`  ·  ` ·  *`  .·•  `·   *·`·   ·*`.

from itertools import permutations

from aoc.aoc_2019.intcode import Computer


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


def _run_until_await_input(computer: Computer) -> None:
    """Runs instructions until the computer halts, or until it requires
    an additional input to continue"""
    
    while True:
        wait = computer.current_instruction == computer.input and len(computer.stdin) == 0
        if computer.halted or wait:
            return
        computer.run_instruction()
    #


def compute_thrust(program: list[int], sequence: tuple[int, ...], feedback=False) -> int:
    """Computes the thruster signal from a chain of amplifiers running an IntCode program.
    sequence: The values used to initialize each computer in the amplifier chain.
    feedback: Whether the output from the final amplifier is fed back into the first"""
    
    # Set up the amplifier circuit
    amplifiers = [Computer(program).add_input(input_) for input_ in sequence]
    signal = 0
    
    while not amplifiers[-1].halted:
        for a in amplifiers:
            a.add_input(signal)
            _run_until_await_input(a)
            signal = a.read_stdout()
        if not feedback:
            break
        #
    return signal


def determine_max_thrust(program, values: tuple[int, ...], use_feedback_loop=False) -> int:
    """Determines the maximum possible thrust signal, using any permutation of
    the input values as the initialization sequence.
    if use_feedback_loop, the signal from the last amplifier is fed back into the first,
    until it halts"""
    n_amplifiers = len(values)
    best = -float("inf")
    for comb in permutations(values, n_amplifiers):
        thrust = compute_thrust(program=program, sequence=comb, feedback=use_feedback_loop)
        if thrust > best:
            best = thrust
        #
    
    assert isinstance(best, int)
    return best


def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)

    star1 = determine_max_thrust(program, values=(0, 1, 2, 3, 4))
    print(f"Solution to part 1: {star1}")

    star2 = determine_max_thrust(program, values=(5, 6, 7, 8, 9), use_feedback_loop=True)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 7
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
