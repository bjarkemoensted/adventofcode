# · .`· ·`*   ·.* ·  `·  .  ·.· `   • ..·*·•.  `· ` ·. ·*   · .    ·*·`·*.   ·*.
# .·*.+·`·    *·`  .· * · · . Some Assembly Required  .`` ·* +·     ·`*.· .· `··
# ·.·   *.· `·  ·      https://adventofcode.com/2015/day/7   ·. *·  ` ·*`·     ·
#  · ·`+ .   ` · *`·.· · * ·`+   .·* · ` · `+ .   · `  . ·*     ·`· • .`· ·*  .`


def parse(s: str):
    """Parses input lines into a, and expression that can be evaluated given the necessary variables and values,
    and b, the output wire in which the resulting output (if it can be computed) should be stored."""
    res = []
    for line in s.split("\n"):
        line_parsed = line
        parts = line.split()
        for part in parts:
            # Replace e.g. 'AND' with '&'
            if part in operators:
                line_parsed = line_parsed.replace(part, operators[part])
            elif part.isdigit():
                bin_ = make_binary(part)
                line_parsed = line_parsed.replace(part, bin_)
            elif part == "->":
                # Stop parsing when we hit the arrow
                break
            else:
                # Make variables on the left side of the arrow formattable
                line_parsed = line_parsed.replace(part, '{'+part+'}')
        expression, output_wire = line_parsed.split(" -> ")
        res.append((expression, output_wire))

    return res


def make_binary(val):
    if isinstance(val, str):
        val = int(val)
    return bin(val & 0b1111111111111111)


def make_decimal(bin_):
    return int(bin_, 2)


operators = {
    'OR': '|',
    'AND': '&',
    'LSHIFT': '<<',
    'RSHIFT': '>>',
    'NOT': '~'
}


def construct_network(instructions, overrides=None):
    """Emulates the wire network. Repeatedly looks for expressions which contain no unknown variables, and
    stores the result in the specified output wire.
    If an expression contains unknowns, we postpone it to the next iteration, hoping that everything can be computed
    at some point."""

    # If any values are hard-coded to a specific values, set them at the beginning
    if overrides is None:
        overrides = {}
    d = {k: v for k, v in overrides.items()}

    operations_remaining = [tup for tup in instructions]
    while operations_remaining:
        next_iteration = []  # Iterations with currently unknown variables. Try again next iteration.
        for expression, output_wire in operations_remaining:
            # Don't update wires that we override
            if output_wire in overrides:
                continue
            # If possible, evaluate the expression using variables we've already figured out
            try:
                signal = eval(expression.format(**d))
                signal_bits = make_binary(signal)
                d[output_wire] = signal_bits
            # Failing that, try again next turn
            except KeyError:
                next_iteration.append((expression, output_wire))
            #
        operations_remaining = next_iteration

    return d


def solve(data: str) -> tuple[int|str, int|str]:
    operations = parse(data)
    signals = construct_network(operations)

    signal_a = signals['a']
    star1 = make_decimal(signal_a)
    print(f"Solution to part 1: {star1}")

    override = {'b': signal_a}
    new_signals = construct_network(operations, override)
    new_signal_a = new_signals['a']

    star2 = make_decimal(new_signal_a)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 7
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()