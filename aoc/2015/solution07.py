# Read in data
with open("input07.txt") as f:
    raw = f.read()

example_input = \
"""123 -> x
456 -> y
x AND y -> d
x OR y -> e
x LSHIFT 2 -> f
y RSHIFT 2 -> g
NOT x -> h
NOT y -> i"""


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


def parse(s):
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


operations = parse(raw)
signals = construct_network(operations)

signal_a = signals['a']
print(f"Signal in wire a: {make_decimal(signal_a)}.")

override = {'b': signal_a}
new_signals = construct_network(operations, override)
new_signal_a = new_signals['a']
print(f"New signal in a: {make_decimal(new_signal_a)}.")
