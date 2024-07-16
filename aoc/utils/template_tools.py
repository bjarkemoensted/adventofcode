from jinja2 import Template

from aoc.utils import config


def read_template():
    with open(config.solution_template_path, "r") as f:
        template = f.read()

    return template


def make_solution_draft(day: int, has_examples=False) -> str:
    raw = read_template()
    temp = Template(raw)

    render_kwargs = dict(
        input_folder=config.input_folder,
        input_filename=config.input_filename.format(day=day)
    )
    if has_examples:
        render_kwargs["example_filename"] = config.example_filename.format(day=day)

    res = temp.render(**render_kwargs)

    return res