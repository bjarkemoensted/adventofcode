from aocd.models import Puzzle
from jinja2 import Template

from aoc.utils import ascii_snow_art, config


def read_template():
    with open(config.solution_template_path, "r") as f:
        template = f.read()

    return template


def make_solution_draft(day: int, year:int) -> str:
    raw = read_template()
    temp = Template(raw)
    header = make_ascii_header(year=year, day=day)

    render_kwargs = dict(
        day=day,
        year=year,
        ascii_header=header
    )

    res = temp.render(**render_kwargs)+"\n"

    return res


def make_ascii_header(year: int, day: int, as_comments=True):
    """Makes an ASCII snow-art header for a new puzzle."""

    line_starter = "# " if as_comments else ""
    min_line_len = 80 - len(line_starter)

    p = Puzzle(year, day)
    elems = (
        p.title,
        p.url
    )

    len_ = max(len(s) for s in elems)
    padding = 3
    n_buffer_lines = 1
    line_length = max(min_line_len, len_ + 2 + padding)
    inject = n_buffer_lines*[""] + list(elems) + n_buffer_lines*[""]

    # Generate the snow-art
    seed = int(f"{year}{day:02}")
    
    let_it_snow = ascii_snow_art.Annealer(seed=seed)
    snow = let_it_snow(rows=len(inject), cols=line_length)

    # Insert the required text
    res_lines = []
    for i, sl in enumerate(snow.splitlines()):
        
        s = inject[i]
        line_chars = list(sl)
        if s != "":
            insert_line = " "+inject[i]+" "
            shift = (len(sl) - len(insert_line)) // 2
            for i, char in enumerate(insert_line):
                line_chars[i + shift] = char
            #
        
        line = line_starter+"".join(line_chars)
        res_lines.append(line)
    
    
    assert len({len(line) for line in res_lines}) == 1
    res = "\n".join(res_lines)
    return res
