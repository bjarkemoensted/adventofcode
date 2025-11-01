import datetime
import pathlib

_here = pathlib.Path(__file__).resolve().parent
root_dir = _here.parents[1]
package_name = "aoc"
solutions_dir = root_dir / package_name

ext_code = ".py"
solution_template_path = _here / "solution.py.template"

solution_filename = "solution{day:02d}"+ext_code
solution_regex    = r"solution(\d{2})"+ext_code
year_regex        = r"(\d{4})"
year_folder       = "aoc_{year:04d}"

_now = datetime.datetime.now()
tis_the_season = _now.month == 12

current_year = _now.year
year_min = 2015
year_max = _now.year if tis_the_season else current_year - 1
years_range = tuple(range(year_min, year_max + 1))

current_day = _now.day
day_min = 1
day_max = 25
days_range = tuple(range(day_min, day_max + 1))


def make_path(year: int, day: int) -> pathlib.Path:
    res = solutions_dir / year_folder.format(year=year) / solution_filename.format(day=day)
    return res


def get_module(year: int, day: int):
    """Gets the module path to use as entry point"""
    
    year_part = year_folder.format(year=year)
    day_part = pathlib.Path(solution_filename.format(day=day)).stem
    res = f"{package_name}.{year_part}.{day_part}"
    return res


if __name__ == '__main__':
    pass
