import argparse
import os
import pathlib

from aoc.utils.template_tools import make_solution_draft
from aoc.utils import config
from aoc.utils.data import read_data_and_examples
from aoc.utils import tokens
from aoc.utils.utils import get_day_and_year


def initialize():
    parser = argparse.ArgumentParser(description="Initialize solution for a new day")
    parser.add_argument(
        "--day",
        "-d",
        type=int,
        default=None,
        help=f"{config.day_min}-{config.day_max} (default: %(default)s)",
    )
    parser.add_argument(
        "--year",
        "-y",
        type=int,
        default=None,
        help=f"{config.year_min}-{config.year_max} (default: %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="create the template file even if source already exists",
    )

    args = parser.parse_args()
    n_none = sum(arg is None for arg in (args.day, args.year))
    if n_none == 1:
        raise ValueError(f"Either specify both day and year, or neither!")

    if n_none > 0:
        day, year = get_day_and_year()
    else:
        day = args.day
        year = args.year

    print(f"Initializing solution for year {year}, day {day}...")
    data, examples = read_data_and_examples(year=year, day=day)
    has_examples = len(examples) > 0

    solution_draft = make_solution_draft(year=year, day=day, has_examples=has_examples)

    _here = pathlib.Path(os.getcwd())

    solution_file = _here / config.solution_filename.format(day=day)
    if pathlib.Path(solution_file).exists() and not args.force:
        print(f"Solution file {solution_file} already exist and --force is not set. Aborting.")

    with open(solution_file, "w") as f:
        f.write(solution_draft)
    #


def fix_tokens():
    parser = argparse.ArgumentParser(description="Interactive CLI token fixing")
    parser.parse_args()
    tokens.fix_tokens()
