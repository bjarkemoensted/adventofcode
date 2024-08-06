import numpy as np
from scipy.optimize import minimize

# Read in data
with open("input15.txt") as f:
    puzzle_input = f.read()

example_input = \
"""Butterscotch: capacity -1, durability -2, flavor 6, texture 3, calories 8
Cinnamon: capacity 2, durability 3, flavor -2, texture -1, calories 3"""


def parse(s):
    """Parse input into list of tuples [('Butterscotch', flavor, etc, ...), ...]"""
    res = {}
    for line in s.split("\n"):
        words = line.split(" ")
        ingredient = words[0].replace(":", "")
        properties = [int(words[i].replace(",", "")) for i in (2, 4, 6, 8, 10)]
        res[ingredient] = properties

    return res


def make_ingredient_matrix(d, include_calories=False):
    """Constructs an 'ingredient matrix' where rows represent each ingredient, and columns represent
    the properties of each (i.e. its 'score' in terms of flavor, texture, etc)."""

    keys = sorted(d.keys())
    n_rows = len(d[keys[0]]) - int(not include_calories)
    shape = (len(keys), n_rows)
    res = np.zeros(shape, dtype=int)

    for i, k in enumerate(keys):
        for j in range(n_rows):
            elem = d[k][j]
            res[i, j] = elem

    return res


def generate_x0(matrix, n_total=100):
    """Given an ingredient matrix, generates a random x0 where optimization procedure can start."""
    rs = np.random.RandomState(seed=42)
    dim = matrix.shape[0]
    res = np.array([0 for _ in range(dim)])
    for _ in range(n_total):
        i = rs.randint(0, dim)
        res[i] += 1

    return res


def make_objective_function(matrix, ignore_last=False):
    """Given an ingredient matrix, generates the function to optimize.
    If ignore_last is set, ignores the rightmost column in the ingredient matrix.
    Can be used to ignore the calories."""

    def f(x):
        res = 1
        for i, column in enumerate(matrix.T):
            last = i == matrix.shape[1] - 1
            if not (last and ignore_last):
                res *= x.dot(column)

        return -res
    return f


def tune_recipe(matrix, loss, constraints):
    """Optimizes the loss function given constraints"""

    x0 = generate_x0(matrix)
    opt = minimize(
        loss,
        x0=x0,
        constraints=constraints,
        method="COBYLA"
    )
    best = opt.x
    return best


data = parse(puzzle_input)
#data = parse(example_input)

M = make_ingredient_matrix(data)
loss = make_objective_function(M)
# Define the constraint that ingredients must sum to 100 (has to be inequalities bc cobyla is weird like that)
constraints = [
    {'type': 'ineq', 'fun': lambda x: sum(x) - 100},
    {'type': 'ineq', 'fun': lambda x: -(sum(x) - 100)},
]

ingredients = tune_recipe(M, loss, constraints)

# Round ingredients to nearest integer and compute score
x = np.array([round(ingredient) for ingredient in ingredients])
score = -loss(x)
print(f"Best score is: {score}.")


# Include calories in ingredients matrix, and define constraints that calories = 500
M2 = make_ingredient_matrix(data, include_calories=True)


def calories(x):
    return x.dot(M2[:, -1])

constraints2 = constraints + [
    {'type': 'ineq', 'fun': lambda x: calories(x) - 500},
    {'type': 'ineq', 'fun': lambda x: -(calories(x) - 500)},
]

# Optimize the new objective
loss2 = make_objective_function(M2, ignore_last=True)
ingredients2 = tune_recipe(M2, loss2, constraints2)

x2 = np.array([round(ingredient) for ingredient in ingredients2])
score2 = -loss2(x2)

print(f"Optimal score with 500 calories: {score2}.")
