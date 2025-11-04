import functools
import importlib
import pathlib
from argparse import ArgumentParser

from aocd.runner import run_for
from aocd.utils import get_plugins

from aoc.utils import config
from aoc.utils.tokens import read_tokens
from aoc.utils.utils import nolog


def disp(s):
    char = "*"
    bound = 3*char
    msg = f"{bound} {s} {bound}"
    print(len(msg)*char)
    print(msg)
    print(len(msg) * char)


def get_available_years_and_days():
    """Returns a dict where keys are the years with available solutions and keys are lists of the days with solutions"""
    d = dict()
    for year in config.years_range:
        days_available = []
        for day in config.days_range:
            yn = config.year_folder.format(year=year)
            dn = pathlib.Path(config.solution_filename.format(day=day)).stem
            try:
                _ = importlib.import_module(f"aoc.{yn}.{dn}")
                days_available.append(day)
            except (ModuleNotFoundError, FileNotFoundError):
                pass
            #
        if days_available:
            d[year] = days_available

    return d


def _run(years: list, days: list, suppress_warnings=True):
    """Wrapper for running against multiple datasets and handling tokens etc."""

    plugs = [ep.name for ep in get_plugins()]
    datasets = read_tokens()

    def _format(label: str, arr: list):
        """Formats stuff like 'years: aoc_2015, 2016', i.e. handles plural s and stuff"""
        formatted = f"{label}{'s' if len(arr) != 1 else ''}: {', '.join(map(str, arr))}"
        return formatted

    # Display which years/days we're running
    s = f"Checking - {_format('Year', years)}. {_format('Day', days)}."
    disp(s)
    
    runner = functools.partial(run_for, plugs=plugs, years=years, days=days, datasets=datasets, timeout=0)

    if suppress_warnings:
        # The AOCD logger outputs warnings every time it waits to get/post
        with nolog():
            res = runner()
            print()
        #
    else:
        res = runner()
    
    return res


def check(years: int|list[int]|None=None, days: int|list[int]|None=None) -> None:
    year2days = get_available_years_and_days()
    if isinstance(years, int):
        years = [years]
    if isinstance(days, int):
        days = [days]

    if years is None:
        years = sorted(year2days.keys())

    days_available_all_years = sorted(set.intersection(*(set(year2days.get(year, [])) for year in years)))
    
    run_years_separately = days is None
    
    if run_years_separately:
        for year in years:
            days_run = year2days[year] if days is None else days
            _run(years=[year], days=days_run)
    else:
        days_run = days_available_all_years if days is None else days
        _run(years=years, days=days_run)
    #


def _parse_days(days_strs: list) -> list|None:
    
    if not days_strs:
        return None

    days = set([])
    
    for s in days_strs:
        parts = s.replace(",", " ").split()
        for part in parts:
            if "-" in part:
                a, b = map(int, part.split("-"))
                addition = set(range(a, b+1))
            else:
                addition = {int(part)}
            if not all(config.day_min <= day <= config.day_max for day in addition):
                raise ValueError(f"Invalid days argument: {part}")
            days |= addition
        
    res = sorted(days)
    return res


def main() -> None:
    parser = ArgumentParser(description="Solution checking CLI tool")

    parser.add_argument(
        "-y",
        "--years",
        metavar=f"({config.year_min}-{config.year_max})",
        type=int,
        nargs="+",
        choices=config.years_range,
        help="Year or years for which to check solutions. Defaults to all available.",
    )
    parser.add_argument(
        "-d",
        "--days",
        metavar=f"({config.day_min}-{config.day_max})",
        type=str,
        nargs="*",
        default=[],
        help="Day or days for which to check solutions. Space separated (1 2 3) including ranges (1 2 5-9 11) works\
        Defaults to check all available for each year.",
    )

    args = parser.parse_args()

    kwargs = args.__dict__
    kwargs["days"] = _parse_days(kwargs["days"])
    check(**kwargs)


if __name__ == '__main__':
    main()
