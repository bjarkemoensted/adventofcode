# .··*  · `·.+ ·     .·     ` •··. `·*·   ·  . `· ·     ·+ ·.  ·*    ··. ` * .·`
# ·  .·.•·   · *  .· ·  ` · *. ·.  Monkey Math   ·*`  . . ·+· .   ·•.  ·  `.*`•·
# ·.   ·* .    `· ·. · https://adventofcode.com/2022/day/21  ··. ·    .` ·+`··.*
# ` .·`*. ·   ·   +`.  ·· .` · +  · * ` .  ·.  · `. ·  .`·   .* ·  .  ` *.··  `.

from copy import deepcopy

import networkx as nx

type instype = tuple[str, str, str]


def parse(s: str) -> dict[str,int|instype]:
    res: dict[str,int|instype] = {}
    for line in s.split("\n"):
        monkey_name, data_string = line.split(": ")
        data = data_string.split(" ")

        if len(data) == 1:
            res[monkey_name] = int(data[0])
        elif len(data) == 3:
            a, op, b = data
            res[monkey_name] = (op, a, b)
        #

    return res


def make_computation_graph(monkey_dict: dict[str,int|instype]) -> nx.DiGraph:
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


def apply_operation(op: str, a: int, b: int) -> float:
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


def compute_value(G: nx.DiGraph, monkey: str) -> float:
    formula = G.nodes[monkey]["formula"]
    operator = formula[0]
    input_monkeys = formula[1:]
    input_values = [G.nodes[input_monkey]["value"] for input_monkey in input_monkeys]
    if any(value is None for value in input_values):
        raise ValueError(f"Missing values in {', '.join(input_monkeys)}.")

    res = apply_operation(operator, *input_values)

    return res


def populate_values(G: nx.DiGraph) -> None:
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
    #


def evaluate_yelling_results(x: int, G: nx.DiGraph) -> float:
    G.nodes["humn"]["value"] = x
    G.nodes["humn"]["formula"] = x
    for node in G.nodes():
        if G.nodes[node]["value"] != G.nodes[node]["formula"]:
            G.nodes[node]["value"] = None
    populate_values(G)
    res = G.nodes["root"]["value"]

    return res


def prepare_graph_for_solving_inplace(G: nx.DiGraph) -> None:
    """Alters computation graph such that the root note computes the difference between its two input nodes.
    Solving the problem amounts to reaching a difference of zero."""

    _, *inputs = G.nodes["root"]["formula"]
    diff_formula = tuple(("-", *inputs))
    G.nodes["root"]["formula"] = diff_formula


def newton_solve(G: nx.DiGraph) -> int:
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
    #


def solve(data: str) -> tuple[int|str, ...]:
    monkey_dict = parse(data)
    G = make_computation_graph(monkey_dict)

    populate_values(G)
    star1 = round(G.nodes["root"]["value"])
    print(f"Solution to part 1: {star1}")

    G = deepcopy(G)
    prepare_graph_for_solving_inplace(G)

    star2 = newton_solve(G)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 21
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
