from collections import defaultdict
from copy import deepcopy
import re



def parse(s):
    assign = []
    swap = []

    for line in s.split("\n"):
        m = re.match(r"bot (\d+) gives low to (bot|output) (\d+) and high to (bot|output) (\d+)", line)
        if m is None:
            m = re.match(r"value (\d+) goes to bot (\d+)", line)
        if m is None:
            raise ValueError(f"Couldn't parse line: {line}.")

        parsed = []
        for elem in m.groups():
            try:
                parsed.append(int(elem))
            except ValueError:
                parsed.append(elem)

        if len(parsed) == 2:
            # Parse assignments like (a, b) means assign value a to bot b
            assign.append(tuple(parsed))
        elif len(parsed) == 5:
            # Parse value swaps like (bot number, (bot/output, index), (...))
            n_bot, reg1, ind1, reg2, ind2 = parsed
            parsed = (n_bot, (reg1, ind1), (reg2, ind2))
            swap.append(parsed)
        else:
            raise ValueError

    return assign, swap


def _make_initial_state(assign):
    state = {}
    for key in ("bot", "output"):
        state[key] = defaultdict(lambda: [])

    for value, bot_number in assign:
        state["bot"][bot_number].append(value)

    return state


def run_instructions(assign, swap, target_comparison):
    """Repeatedly has bots holding 2 microchips pass them along according to the input rules.
    target_comparison denotes which comparison of chips we're after, so if target_comparison=[42, 60],
    the method will return the number of the robot which compares chips 42 and 60, along with the final state
    of bots and outputs."""

    target_comparison = sorted(target_comparison)
    instructions = {bot: (low, high) for bot, low, high in swap}
    state = _make_initial_state(assign)
    bot_making_target_comparison = None

    loaded = [bot for bot in state["bot"].keys() if len(state["bot"][bot]) == 2]
    while loaded:
        new_state = deepcopy(state)
        for bot in loaded:
            del new_state["bot"][bot]
        for bot in loaded:
            values = sorted(state["bot"][bot])
            if values == target_comparison:
                bot_making_target_comparison = bot

            for value, rule in zip(values, instructions[bot]):
                reg, ind = rule
                new_state[reg][ind].append(value)
            #

        state = new_state
        loaded = [bot for bot in state["bot"].keys() if len(state["bot"][bot]) == 2]

    return state, bot_making_target_comparison


def solve(data: str, chips=None):
    if chips is None:
        chips = [17, 61]
    instructions = parse(data)

    end_state, bot = run_instructions(*instructions, target_comparison=chips)
    star1 = bot
    print(f"Solution to part 1: {star1}")

    solution2 = 1
    for n in (0, 1, 2):
        solution2 *= end_state["output"][n][0]

    star2 = solution2
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2016, 10
    from aoc.utils.data import check_examples
    extra_kwargs_parser = lambda s: dict(chips=[int(m) for m in re.findall(r"value-(\d+)", s)])
    check_examples(year=year, day=day, solver=solve, extra_kwargs_parser=extra_kwargs_parser)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
