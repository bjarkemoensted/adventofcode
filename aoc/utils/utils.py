from contextlib import contextmanager
import logging

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


def get_day_and_year():
    """Attempts to automatically determine the day and year for the next solution to attempt.
    Looks at existing solution files and tries to determine the next day from those.
    Attempts to determine the year from the solution files and their parent directories."""

    if config.tis_the_season and config.current_day in config.days_range:
        return config.current_day, config.current_year
    
    raise RuntimeError("Could not automatically determine year and day.")
