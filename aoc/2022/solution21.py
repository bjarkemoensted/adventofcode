from copy import deepcopy
import networkx as nx


def read_input():
    with open("input21.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


test = """root: pppw + sjmn
dbpl: 5
cczh: sllz + lgvd
zczc: 2
ptdq: humn - dvpt
dvpt: 3
lfqf: 4
humn: 5
ljgn: 2
sjmn: drzm * dbpl
sllz: 4
pppw: cczh / lfqf
lgvd: ljgn * ptdq
drzm: hmdt - zczc
hmdt: 32"""


def parse(s):
    res = {}
    for line in s.split("\n"):
        monkey_name, data_string = line.split(": ")
        data = data_string.split(" ")
        formula = None
        if len(data) == 1:
            value = int(data[0])
            formula = value
        elif len(data) == 3:
            a, op, b = data
            formula = (op, a, b)

        res[monkey_name] = formula
    return res


def make_computation_graph(monkey_dict):
    """Makes a DAG representing the monkey dependencies.
    Each node corresponds to a monkey and has a 'value' field denoting the value it should yell (initialized as None).
    Each monkey also has a 'formula' field which is an integer if the monkey starts out shouting a number, otherwise
    a tuple like (monkey_a, monkey_b, '+') if the monkey should yell e.g. the sum of monkeys a and b.
    Monkey a maps to monkey b iff monkey b's number depends on monkey a."""

    G = nx.DiGraph()

    # Add nodes
    for monkey, data in monkey_dict.items():
        props = dict(formula=data, value=None)
        G.add_node(monkey, **props)

    # Add edges for dependencies
    for monkey, data in monkey_dict.items():
        if isinstance(data, int):
            G.nodes[monkey]["value"] = data
        elif isinstance(data, tuple):
            for dependent_on in data[1:]:
                G.add_edge(dependent_on, monkey)
            #
        #

    return G


def apply_operation(op, a, b):
    """Returns a <operator> b."""

    funs = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x*y,
        "/": lambda x, y: x / y
    }

    fun = funs[op]
    res = fun(a, b)
    return res


def compute_value(G, monkey):

    formula = G.nodes[monkey]["formula"]
    operator = formula[0]
    input_monkeys = formula[1:]
    input_values = [G.nodes[input_monkey]["value"] for input_monkey in input_monkeys]
    if any(value is None for value in input_values):
        raise ValueError(f"Missing values in {', '.join(input_monkeys)}.")

    res = apply_operation(operator, *input_values)
    return res


def populate_values(G):
    # Start with the 'constant nodes' which take no inputs
    active_set = {node for node in G.nodes() if G.nodes[node]["value"] == G.nodes[node]["formula"]}

    while active_set:
        new_set = set([])
        for monkey in active_set:
            for other_monkey in G[monkey]:
                try:
                    value = compute_value(G, other_monkey)
                    G.nodes[other_monkey]["value"] = value
                    new_set.add(other_monkey)
                except ValueError:
                    pass
                #
            #
        active_set = new_set

    return


def evaluate_yelling_results(x, G):
    G.nodes["humn"]["value"] = x
    G.nodes["humn"]["formula"] = x
    for node in G.nodes():
        if G.nodes[node]["value"] != G.nodes[node]["formula"]:
            G.nodes[node]["value"] = None
    populate_values(G)
    res = G.nodes["root"]["value"]

    return res


def prepare_graph_for_solving_inplace(G):
    """Alters computation graph such that the root note computes the difference between its two input nodes.
    Solving the problem amounts to reaching a difference of zero."""

    operator, *inputs = G.nodes["root"]["formula"]
    diff_formula = tuple(("-", *inputs))
    G.nodes["root"]["formula"] = diff_formula


def newton_solve(G):
    """Employs a Newton–Raphson type solving algorithm to find the value at humn node which results in root node
    getting value 0 (corresponding to its input monkeys yelling the same number)"""

    x0 = G.nodes["humn"]["value"]

    while True:
        # Evaluate at current x
        f0 = evaluate_yelling_results(x0, G)
        if f0 == 0:
            return x0

        # Find the gradient at current value of x
        x_shifted = x0 + 1
        f1 = evaluate_yelling_results(x_shifted, G)
        gradient = (f1 - f0)/(x_shifted - x0)

        # Update x according to the Newton method: x0 -> x0 - f(x0)/f'(x0)
        increment = f0/gradient
        x0 -= round(increment)


def main():
    raw = read_input()
    monkey_dict = parse(raw)
    G = make_computation_graph(monkey_dict)

    populate_values(G)
    root_val = round(G.nodes["root"]["value"])
    print(f"Root monkey yells: {root_val}.")

    G = deepcopy(G)
    prepare_graph_for_solving_inplace(G)

    yell = newton_solve(G)
    print(f"The correct number to yell is {round(yell)}.")


if __name__ == '__main__':
    main()
