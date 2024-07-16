import argparse
import json
import os
import pathlib

from aoc.utils.template_tools import make_solution_draft
from aoc.utils import config
from aoc.utils import crypto
from aoc.utils.data import read_data
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
    data, examples = read_data(year=year, day=day)
    has_examples = len(examples) > 0

    solution_draft = make_solution_draft(day=day, has_examples=has_examples)

    _here = pathlib.Path(os.getcwd())
    input_dir = _here / config.input_folder
    input_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_dir / config.input_filename.format(day=day)
    with open(input_file, "w") as f:
        f.write(data)

    if has_examples:
        example_file = input_dir / config.example_filename.format(day=day)
        with open(example_file, "w") as f:
            json.dump(examples, f, sort_keys=True, indent=4)
        #

    box = crypto.Box()
    box.encrypt(input_file)
    box.encrypt(example_file, overwrite_if_exists=True)

    solution_file = _here / config.solution_filename.format(day=day)
    if pathlib.Path(solution_file).exists() and not args.force:
        print(f"Solution file {solution_file} already exist and --force is not set. Aborting.")

    with open(solution_file, "w") as f:
        f.write(solution_draft)
