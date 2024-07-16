import datetime
import json
import pathlib


_here = pathlib.Path(__file__).resolve().parent
root_dir = _here.parents[2]

_input_file_pattern = "input*"
ext_data = ".txt"
ext_encrypted = ".dat"
ext_code = ".py"
solution_template_path = _here / "solution_template.py"

solution_filename = "solution{day:02d}"+ext_code
solution_regex    = "solution(\d{2})"+ext_code
year_regex        = "(\d{4})"
input_filename    = "input{day:02d}"+ext_data
example_filename  = "example{day:02d}"+ext_data
input_folder      = "inputs"

day_min = 1
day_max = 25
days_range = tuple(range(day_min, day_max + 1))

_now = datetime.datetime.now()
year_min = 2015
year_max = _now.year - int(_now.month != 12)
years_range = tuple(range(year_min, year_max + 1))

_secrets = _here / "secrets.json"
def read_secrets():
    try:
        with open(_secrets) as f:
            d = json.load(f)
        #
    except FileNotFoundError:
        d = dict()

    return d


def update_secrets(overwrite=False, **kwargs):
    d = read_secrets()
    overlap = sorted(set(d.keys()) & set(kwargs.keys()))
    if overlap and not overwrite:
        raise ValueError(f"The following keys already exist in the secrets file: {', '.join(overlap)}.")

    for k, v in kwargs.items():
        d[k] = v

    with open(_secrets, "w") as f:
        json.dump(d, f, indent=4, sort_keys=True)


salt = b'\xff\xb5L6\x87\\\x88\xbf\xf4\xcaw\xfau\xda\xbd\xd5'
password_field = 'password'
