from contextlib import contextmanager
import glob
import logging
import os
import pathlib
import re

from aoc.utils import config


@contextmanager
def nolog():
    previous_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_level)
    #



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

    if config.tis_the_season and config.current_day in config.days_range:
        return config.current_day, config.current_year
    
    raise RuntimeError("Could not automatically determine year and day.")



if __name__ == '__main__':
    pass
