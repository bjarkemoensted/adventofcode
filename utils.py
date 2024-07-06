#!/usr/bin/env python3

import argparse
import json
from nacl import pwhash, secret
import os
from pathlib import Path
import re
import sys


_here = os.path.dirname(os.path.abspath(__file__))

_ext_code = ".py"
_ext_data = ".txt"
_ext_encrypted = ".dat"
_input_file_pattern = "input*"
_secret_file = os.path.join(_here, "secrets.json")

_solution_template_file = os.path.join(_here, 'solution_template.py')

################## Templating stuff ###################################


def prompt_yn(prompt:str=None, default="n", allow_non_yes_no_answers=False):
    if default not in "yn":
        raise ValueError

    yn_prompt = f"Continue ({'Y/n' if default == 'y' else 'y/N'})? "
    prompt = yn_prompt if not prompt else f"{prompt} {yn_prompt}"
    input_ = input(prompt).strip()

    x = input_.lower()
    if not input_:
        return default == "y"
    if x == "y":
        return True
    elif x == "n":
        return False
    elif allow_non_yes_no_answers:
        return input_
    else:
        raise ValueError(F"Invalid input: {input_}.")




def _read_template():
    here = os.path.dirname(os.path.realpath(__file__))
    fn = os.path.join(here, _solution_template_file)
    with open(fn, "r") as f:
        template = f.read()

    return template


def initialize(day="auto"):
    # Determine filename
    paths = [path for path in Path(os.getcwd()).glob("*.py")]
    if day == "auto":
        latest = max(map(int, sum([re.findall("solution(\d\d)"+_ext_code, path.name) for path in paths], [])))
        day = latest + 1
    else:
        day = int(day)

    prompt = f"About to initialize for day {day}. Type y/n to proceed, or enter another day to initialize"
    response = prompt_yn(prompt, default="y", allow_non_yes_no_answers=True)

    if response is False:
        print("Aborting...")
        return
    elif isinstance(response, str):
        day = int(response)

    daymin, daymax = 1, 25
    if not (daymin <= day <= daymax):
        raise ValueError(f"Can't initialize for day {day}. Must be between {daymin} and {daymax}")

    solution_filename = f"solution{str(day).zfill(2)}.py"
    input_filename = os.path.join("input", f"input{str(day).zfill(2)}.txt")

    if os.path.exists(solution_filename):
        overwrite = prompt_yn(f"{solution_filename} already exists. Overwrite?", default="n")
        if not overwrite:
            print("Aborting")
            return
        #

    # Build the python solution file from template
    content = _read_template()
    content = content.format(
        INPUT_FILENAME=input_filename
    )

    with open(solution_filename, "w") as f:
        f.write(content)
    print(f"Wrote content to {solution_filename}.")

    # Make sure the folder for input files exists
    input_dir = os.path.dirname(input_filename)
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    with open(input_filename, "w") as f:
        f.write("Insert puzzle input here")
    print(f"Wrote content to {input_filename}.")

    print("All done!")


################## Encryption stuff ###################################


def _salt():
    salt = b'\xff\xb5L6\x87\\\x88\xbf\xf4\xcaw\xfau\xda\xbd\xd5'
    return salt


def _read_password():
    _bad = "SECRET_PASSWORD_HERE"
    try:
        with open(_secret_file, "r") as f:
            d = json.load(f)
        pw = d["password"]
    except FileNotFoundError:
        pw = None



    if pw is None or pw == _bad:
        raise ValueError(f"Put a secret string in {_secret_file}.")

    res = pw.encode()
    return res


def _box_from_password(password: bytes=None):
    if password is None:
        password = _read_password()
    kdf = pwhash.argon2i.kdf
    salt = _salt()
    key = kdf(secret.SecretBox.KEY_SIZE, password, salt=salt)
    box = secret.SecretBox(key)
    return box


def _apply_encryption_op(
        ext_from: str,
        ext_to:str,
        operation: str,
        overwrite_if_exists=False,
        confirm=True
    ):

    to_from_pairs = []
    glob_pattern = _input_file_pattern+ext_from
    folder = _here
    for infile in Path(folder).rglob(glob_pattern):
        file_base, _ = os.path.splitext(infile)
        outfile = file_base+ext_to
        if overwrite_if_exists or not os.path.exists(outfile):
            to_from_pairs.append((infile, outfile))
        #

    if not to_from_pairs:
        print("Nothing to do...")
        return

    print(f"About to {operation} {len(to_from_pairs)} files matching {glob_pattern} in {folder}.")
    if confirm:
        if input("Continue (Y/n)? ").strip() not in ["y", "Y", ""]:
            print("Aborting...")
            return
        #

    box = _box_from_password()
    fun = getattr(box, operation)
    for fn_in, fn_out in to_from_pairs:
        print(f"{operation.title()}ing: {os.path.relpath(fn_in, _here)} -> {os.path.relpath(fn_out, _here)}")
        with open(fn_in, "rb") as fin:
            content = fin.read()
        with open(fn_out, "wb") as fout:
            new_content = fun(content)
            fout.write(new_content)
        #

    print("Done")


def encrypt(overwrite_if_exists = False, confirm = True):
    _apply_encryption_op(
        ext_from=_ext_data,
        ext_to=_ext_encrypted,
        operation="encrypt",
        overwrite_if_exists=overwrite_if_exists,
        confirm=confirm
    )


def decrypt(overwrite_if_exists = False, confirm = True):
    _apply_encryption_op(
        ext_from=_ext_encrypted,
        ext_to=_ext_data,
        operation="decrypt",
        overwrite_if_exists=overwrite_if_exists,
        confirm=confirm
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    # create the top-level parser
    command_varname = "command"
    parser = argparse.ArgumentParser(prog=command_varname)
    subparsers = parser.add_subparsers(dest=command_varname, help='sub-command help')

    # create the parser for the "a" command
    parser_initialize = subparsers.add_parser(
        'initialize',
        help='Initialize a solution + input file.'
    )

    parser_initialize.add_argument(
        '--day',
        '-d',
        help="The day for which to initialize a solution python file. Defaults to inferring date.",
        type=str,
        default="auto"
    )

    parser_encrypt = subparsers.add_parser('encrypt', help='Encrypts alle input files.')
    parser_decrypt = subparsers.add_parser('decrypt', help='Decrypts all input files')
    for subparser in (parser_encrypt, parser_decrypt):
        subparser.add_argument(
            '--overwrite_if_exists',
            type=bool,
            help='Whether to overwrite existing files',
            default=False
        )
        subparser.add_argument(
            '--confirm',
            type=bool,
            help='Whether to prompt user before writing data',
            default=True
        )

    parsed = parser.parse_args()
    command = getattr(parsed, command_varname)
    kwargs = {k: v for k, v in vars(parsed).items() if k != command_varname}
    return command, kwargs


def main(command, kwargs):
    thismodule = sys.modules[__name__]
    fun = getattr(thismodule, command)
    return fun(**kwargs)


if __name__ == '__main__':
    command, kwargs = parse_arguments()
    main(command, kwargs)
