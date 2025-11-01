import importlib
import os
from contextlib import redirect_stdout

from aoc.utils import config


def solve(year, day, data):
    module = config.get_module(year=year, day=day)

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            solution_module = importlib.import_module(module)
            solution = solution_module.solve(data)

    return solution
