from contextlib import redirect_stdout
import importlib
import os
import pathlib

import aoc
from aoc.utils import config


def solve(year, day, data):
    yn = config.year_folder.format(year=year)
    dn = pathlib.Path(config.solution_filename.format(day=day)).stem

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            solution_module = importlib.import_module(f"aoc.{yn}.{dn}")
            solution = solution_module.solve(data)

    return solution


if __name__ == "__main__":
    from aoc.utils.data import read_data_and_examples
    year_ = 2017
    day_ = 1
    data, _ = read_data_and_examples(year=year_, day=day_)
    res = solve(year=year_, day=day_, data=data)
    print(res)
