import importlib
import pathlib

from aoc.utils import config


def get_available_years_and_days():
    d = dict()
    for year in config.years_range:
        days_available = []
        for day in config.days_range:
            yn = config.year_folder.format(year=year)
            dn = pathlib.Path(config.solution_filename.format(day=day)).stem
            try:
                _ = importlib.import_module(f"aoc.{yn}.{dn}")
                days_available.append(day)
            except ModuleNotFoundError:
                pass
            #
        if days_available:
            d[year] = days_available

    return d


def run_for_all(years=None, days=None):
    from aoc.utils.data import read_tokens
    from aocd.utils import get_plugins

    year2days = get_available_years_and_days()
    if years is None:
        years = sorted(year2days.keys())
    if days is None:
        for year in years:
            these_days = set(year2days.get(year, []))
            days = these_days if days is None else days.intersection(these_days)
        days = sorted(days)

    print(f"Running - Years: {years}. Days: {days}.")

    plugs = [ep.name for ep in get_plugins()]
    datasets = read_tokens()
    from aocd.runner import run_for

    #TODO address rate limiting here
    run_for(plugs=plugs, years=years, days=days, datasets=datasets)


if __name__ == '__main__':
    run_for_all()
