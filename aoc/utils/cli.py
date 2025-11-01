import argparse
import pathlib

from aoc.utils import config, tokens
from aoc.utils.template_tools import make_solution_draft
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
        raise ValueError("Either specify both day and year, or neither!")

    if n_none > 0:
        day, year = get_day_and_year()
    else:
        day = args.day
        year = args.year

    print(f"Initializing solution for year {year}, day {day}...")
    solution_draft = make_solution_draft(year=year, day=day)

    path = config.make_path(year=year, day=day)
    path.parent.mkdir(parents=True, exist_ok=True)

    if pathlib.Path(path).exists() and not args.force:
        print(f"Solution file {path} already exist and --force is not set. Aborting.")
        return

    with open(path, "w") as f:
        f.write(solution_draft)
    #


def fix_tokens():
    parser = argparse.ArgumentParser(description="Interactive CLI token fixing")
    parser.parse_args()
    tokens.fix_tokens()
