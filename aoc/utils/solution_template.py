import pathlib
_here = pathlib.Path(__file__).resolve().parent
{% if example_filename -%}
import json
{% endif %}


def read_input():
    fn = _here / "{{input_folder}}" / "{{input_filename}}"
    with open(fn) as f:
        puzzle_input = f.read()
    return puzzle_input


{% if example_filename -%}
def read_examples():
    fn = _here / "{{input_folder}}" / "{{example_filename}}"
    with open(fn) as f:
        d = json.load(f)
    return d
{% endif %}


def parse(s):
    res = s  # TODO parse input here
    return res


def solve():
    raw = read_input()
    parsed = parse(raw)

    # TODO solve puzzle
    star1 = None
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


if __name__ == '__main__':
    solve()
