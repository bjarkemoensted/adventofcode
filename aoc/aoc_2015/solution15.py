from collections import Counter
from functools import reduce
import math
import numpy as np
from operator import mul
from scipy.optimize import minimize


def parse(s):
    res = dict()
    for line in s.splitlines():
        ingredient, mess = line.split(": ")
        d = dict()
        for part in mess.split(", "):
            prop, val = part.split(" ")
            val = int(val)
            d[prop] = val
        res[ingredient] = d
    return res


def _constraints_satisfied(x: np.array, constraints: list|dict) -> bool:
    if isinstance(constraints, dict):
        fun = constraints["fun"]
        type_ = constraints["type"]
        if type_ == "ineq":
            res = fun(x) >= 0
        elif type_ == "eq":
            res = fun(x) == 0
        else:
            raise ValueError
        return res
    else:
        return all(_constraints_satisfied(x, c) for c in constraints)
    #


def make_x0(n_ingredients: int, total_amount: int, loss, seed: int = None, constraints=None):
    if constraints is None:
        constraints = []

    rs = np.random.RandomState(seed=seed)
    inds = list(range(n_ingredients))
    while True:
        counts = Counter(rs.choice(a=inds, size=total_amount, replace=True))
        x0 = np.array([counts.get(i, 0) for i in inds])
        if loss(x0) == 0:
            continue
        if _constraints_satisfied(x0, constraints):
            return x0
        #
    #


def iter_combs(values, n):
    for val in values:
        head = [val]
        if n == 1:
            yield head
        else:
            for sublist in iter_combs(values, n-1):
                yield head + sublist
            #
        #
    #


def jitter(x, loss, constraints):
    shifts = [-1, 0, 1]
    record = float("inf")
    best = None
    n = 0

    for diff in iter_combs(values=shifts, n=len(x)):
        n += 1
        new_x = np.array([math.floor(val + delta) for val, delta in zip(x, diff)])

        if not _constraints_satisfied(new_x, constraints):
            continue
        new_loss = loss(new_x)
        if new_loss < record:
            record = new_loss
            assert loss(new_x) == record
            best = new_x
            assert loss(best) == record


    print(record)
    assert n == len(shifts)**len(x)

    return best


def opt(ingredients, total_amount=100, n_calories: int = None):
    ing_ordered = sorted(ingredients.keys())
    all_props = set(list(ingredients.values())[0].keys())
    assert all(set(v.keys()) == all_props for v in ingredients.values())
    cal = "calories"
    all_props.remove(cal)
    props_order = sorted(all_props)

    W = np.array([[ingredients[ing][prop] for prop in props_order] for ing in ing_ordered])
    n_ingredients, n_properties = W.shape

    loss = lambda x_: -reduce(mul, (max(val, 0) for val in W.T.dot(x_)), 1)

    constraints = [
        {'type': 'ineq', 'fun': lambda x_, fac=sig: fac*(sum(x_) - total_amount)} for sig in (+1, -1)
        #{'type': 'eq', 'fun': lambda x_: (sum(x_) - total_amount)}
    ]
    for i, col in enumerate(W):
        constraints.append(
            {'type': 'ineq', 'fun': lambda x_, ind=i: sum(x_[ind]*v for v in col)}
        )

    cals = [ingredients[ing][cal] for ing in ing_ordered]

    print(len(constraints))
    if isinstance(n_calories, int):
        print(f"{cals=}")
        w_cal = np.array([ingredients[ing][cal] for ing in ing_ordered])
        constraints += [
            {'type': 'ineq', 'fun': lambda x_, fac=sig: fac*(sum(val * c for val, c in zip(x_, cals)) - n_calories)}
            for sig in (+1, -1)
            #{'type': 'ineq', 'fun': lambda x_: sig * (w_cal.T.dot(x_) - n_calories} for sig in (+1, -1)
            #{'type': 'eq', 'fun': lambda x_: w_cal.T.dot(x_) - n_calories}
        ]
        print(n_calories, len(constraints))

    bounds = [(0, None) for _ in range(n_ingredients)]
    x0 = make_x0(n_ingredients=n_ingredients, total_amount=total_amount, loss=loss, seed=42, constraints=constraints)

    res = minimize(
        loss,
        x0,
        #method='SLSQP',
        method='COBYLA',
        #bounds=bounds,
        constraints=constraints,
        options=dict(maxiter=99999)
    )

    # print(f"{res=}")
    # print(_constraints_satisfied(res.x, constraints))
    res_float = res.x
    res_int = jitter(x=res_float, loss=loss, constraints=constraints)
    if res_int is None:
        raise RuntimeError


    # print(f"{x0=}", f"{_constraints_satisfied(x0, constraints)}")
    # print(f"{bounds=}")
    # print(f"{constraints=}")
    # print(f"{loss=}")
    # print(f"{W=}")
    # print()
    # print(f"{res_float=}")
    # print(f"{res_int=}")

    print(f"Calories: {sum(a*b for a, b in zip(res_int, cals))} ({sum(a*b for a, b in zip(res_float, cals))})")

    score = -loss(res_int)
    return score


def solve(data: str):
    ingredients = parse(data)

    # 13882464
    # 13882464
    star1 = opt(ingredients)

    print(f"Solution to part 1: {star1}")

    # 11171160 ?
    star2 = opt(ingredients, n_calories=500)  # 117936
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2015, 15
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
