#  .·`• ··.     ·*  *`    ·+·*`. · *·`     ·`  •·   `·. *·`.*· +.`·     .·*+`.··
# .·*`· .`·.  *`   .    ·   Science for Hungry People * +.·  `·  *  `·      *·.`
# ·` · .+  `* ·· `*  · https://adventofcode.com/2015/day/15   .` .*·+`·    .·` .
# ·.· `+.·`•·` * ·.    ·     ·  · * .·· •` . . · ·*· .  ·    +`  · `·*.·`+·` *` 


import math
from collections import Counter
from functools import reduce
from operator import mul

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


def parse(s: str):
    res = dict()
    for line in s.splitlines():
        ingredient, mess = line.split(": ")
        d = dict()
        for part in mess.split(", "):
            prop, val_str = part.split(" ")
            val = int(val_str)
            d[prop] = val
        res[ingredient] = d
    return res


def _constraints_satisfied(x: NDArray[np.int_], constraints: list|dict) -> bool:
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


def make_x0(n_ingredients: int, total_amount: int, loss, seed: int|None = None, constraints=None):
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


def opt(ingredients, total_amount=100, n_calories: int|None = None):
    ing_ordered = sorted(ingredients.keys())
    all_props = set(list(ingredients.values())[0].keys())
    assert all(set(v.keys()) == all_props for v in ingredients.values())
    cal = "calories"
    all_props.remove(cal)
    props_order = sorted(all_props)

    W = np.array([[ingredients[ing][prop] for prop in props_order] for ing in ing_ordered])
    n_ingredients, _ = W.shape

    def loss(x_):
        return -reduce(mul, (max(val, 0) for val in W.T.dot(x_)), 1)

    constraints = [
        {'type': 'ineq', 'fun': lambda x_, fac=sig: fac*(sum(x_) - total_amount)} for sig in (+1, -1)
    ]
    for i, col in enumerate(W):
        constraints.append(
            {'type': 'ineq', 'fun': lambda x_, ind=i: sum(x_[ind]*v for v in col)}
        )

    cals = [ingredients[ing][cal] for ing in ing_ordered]

    if isinstance(n_calories, int):
        constraints += [
            {'type': 'ineq', 'fun': lambda x_, fac=sig: fac*(sum(val * c for val, c in zip(x_, cals)) - n_calories)}
            for sig in (+1, -1)
        ]

    x0 = make_x0(n_ingredients=n_ingredients, total_amount=total_amount, loss=loss, seed=42, constraints=constraints)

    res = minimize(
        loss,
        x0,
        method='COBYLA',
        constraints=constraints,
        options=dict(maxiter=99999)
    )

    res_float = res.x
    res_int = jitter(x=res_float, loss=loss, constraints=constraints)
    if res_int is None:
        raise RuntimeError

    print(f"Calories: {sum(a*b for a, b in zip(res_int, cals))} ({sum(a*b for a, b in zip(res_float, cals))})")

    score = -loss(res_int)
    return score


def solve(data: str) -> tuple[int|str, int|str]:
    ingredients = parse(data)

    star1 = opt(ingredients)

    print(f"Solution to part 1: {star1}")

    star2 = opt(ingredients, n_calories=500)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 15
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
