#!/usr/bin/env python3
"""Helper script to set up a file for an Advent of Code solution.
Can be called with e.g. python initialize_solution.py 4 to create a python and an input file for December 4th.
Can also be called with --day=4.
'overwrite' is an optional argument, denoting whether existing files may be overwritten. Defaults to prompting user."""

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--day', '-d', help="The day for which to initialize a solution python file.", type=int)
parser.add_argument('--overwrite', '-ow', help="Whether to overwrite existing file", type=bool, default=None)


def read_template():
    here = os.path.dirname(os.path.realpath(__file__))
    fn = os.path.join(here, 'solution_template.py')
    with open(fn, "r") as f:
        template = f.read()

    return template


def parse_args():
    """Parses the arguments from stdin.
    Returns in keyword format {day: int, overwrite: bool}"""

    args, unknownargs = parser.parse_known_args()
    day = args.day
    if day is None:
        try:
            day = int(unknownargs[0])
        except ValueError:
            pass
        #
    if day is None or not isinstance(day, int) or not (1 <= day <= 25):
        raise RuntimeError("No day specified")

    res = dict(day=day, overwrite=args.overwrite)
    return res


def make_input_filename(day: int):
    res = f"input{str(day).zfill(2)}.txt"
    return res


def make_solution_filename(day: int):
    res = f"solution{str(day).zfill(2)}.py"
    return res


def prompt_overwrite(files):
    """Prompt user if files may be overwritten"""
    if len(files) == 1:
        base = f"File {files[0]} already exists."
    else:
        base = f"Files {', '.join(sorted(files))} already exist."

    prompt = base+" Overwrite? (y/N)"
    user_input = input(prompt)
    overwrite = user_input in ("y", "Y", "yes")
    return overwrite


def go(day, overwrite=None):
    """Sets up a python file for solving an Advent of Code problem for the specified day, with functionality for
    reading puzzle input etc."""

    # Determine filename
    solution_filename = make_solution_filename(day)
    input_filename = make_input_filename(day)

    # Build the python solution file
    content = read_template()
    content = content.format(
        INPUT_FILENAME=input_filename
    )

    # Determine whether to save files
    write_file = True
    existing_files = [fn for fn in (solution_filename, input_filename) if os.path.isfile(fn)]
    if existing_files:
        # If it hasn't been specified whether files may be overwritten, prompt the user
        if overwrite is None:
            write_file = prompt_overwrite(existing_files)
        else:
            write_file = overwrite

    if write_file:
        with open(solution_filename, "w") as f:
            f.write(content)
        print(f"Wrote content to {solution_filename}.")

        with open(input_filename, "w") as f:
            f.write("Insert puzzle input here")
        print(f"Wrote content to {input_filename}.")
    else:
        print("Nothing to do...")


if __name__ == '__main__':
    kwargs = parse_args()
    go(**kwargs)
