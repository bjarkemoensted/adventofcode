import glob
import os
import pathlib
import re

from aoc.utils import config


def _day_year_valid(day: int = None, year: int = None) -> bool:
    """Whether a year and/or day is valid"""
    if all(arg is None for arg in (day, year)):
        raise ValueError

    arg_lim_tuples = [
        (day, config.day_min, config.day_max),
        (year, config.year_min, config.year_max)
    ]
    for arg, min_, max_ in arg_lim_tuples:
        if arg is None:
            continue
        if not (min_ <= arg <= max_):
            return False
        #
    return True


def _infer_year_day_from_filename(fn, n_parents=3):
    """Attempts to infer the year and day from a filename and its n parent dir(s)."""

    # Represent the file and containing dir(s) as a string in which we can search for patterns matching day/year formats
    path = pathlib.Path(fn)
    parts = path.parts[-n_parents:]
    path_truncated = ";".join(parts)

    # Attempt matching
    day_matches = re.findall(config.solution_regex, path_truncated)
    year_matches = re.findall(config.year_regex, path_truncated)
    if any(len(hits) == 0 for hits in (day_matches, year_matches)):
        return None
    day = int(day_matches[-1])
    year = int(year_matches[-1])

    # Make sure values are within bounds
    if _day_year_valid(day=day, year=year):
        return year, day
    else:
        return None


def get_day_and_year():
    """Attempts to automatically determine the day and year for the next solution to attempt.
    Looks at existing solution files and tries to determine the next day from those.
    Attempts to determine the year from the solution files and their parent directories."""

    cwd = os.getcwd()
    code_files = glob.glob(os.path.join(cwd, f"*{config.ext_code}"))

    year_day_tuples = []
    for fn in code_files:
        yd = _infer_year_day_from_filename(fn)
        if yd is not None:
            year_day_tuples.append(yd)
        #

    year = day = None
    if year_day_tuples:
        # If year and day could be inferred form solution files, bump by 1 day
        year, day = max(year_day_tuples)
        day += 1
    else:
        try:
            # Otherwise, attempt to determine year and just go with the first possible day (e.g. if starting a new year)
            year = int(re.findall(config.year_regex, str(cwd))[-1])
            day = config.day_min
        except IndexError:
            pass
        #

    # Throw an error on invalid values for year/day, or if neither could be determined
    if any(val is None for val in (day, year)) or not _day_year_valid(day=day, year=year):
        raise RuntimeError("Could not automatically determine year and day.")

    return day, year
