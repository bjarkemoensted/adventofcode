from aocd.cookies import get_working_tokens
from aocd.utils import get_owner
from aocd.exceptions import DeadTokenError
from aocd.models import Puzzle, User, default_user, AOCD_CONFIG_DIR

import dis
from hashlib import md5
import json
import sys
from termcolor import cprint
from typing import Callable

from aoc.utils.temp import TempCache


def read_data_and_examples(year: int, day: int, token=None):
    user = User(token=token) if token is not None else default_user()
    puzzle = Puzzle(year=year, day=day, user=user)

    data = puzzle.input_data
    examples = [ex._asdict() for ex in puzzle.examples]

    return data, examples


def _hash_callable(f: Callable) -> str:
    instructions = dis.Bytecode(f).dis()
    res = md5(instructions.encode("utf8")).hexdigest()
    return res


def solver_looks_new(solver: Callable) -> bool:
    hash_ = _hash_callable(solver)
    fn = ".aoc_solver_hash_cache.txt"
    looks_new = True
    try:
        with TempCache(fn, "r") as f:
            seen = set(f.read().splitlines())
        looks_new = hash_ not in seen
        #
    except FileNotFoundError:
        seen = set([])

    if looks_new:
        with TempCache(fn, "a") as f:
            if seen:
                f.write("\n")
            f.write(hash_)
        #
    return looks_new


def _coerce_val(val) -> str:
    return str(val)


class StreamFilter:
    def __init__(self, filter_: str):
        self.filter_ = filter_
        self.stream = None
        self.just_ignored = False
        self._old_stdout = None

    def write(self, s):
        ignore = (self.filter_ is not None) and s.startswith(self.filter_)
        if ignore:
            self.just_ignored = True
            return
        if self.just_ignored and s == "\n":
            self.just_ignored = False
        else:
            self.stream.write(s)
        #

    def __enter__(self):
        self.stream = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.stream


def _evaluate_examples(solver: Callable, examples: list, suppress_output: bool, extra_kwargs_parser=None):
    """Takes a solver and list of examples (as exposed in the aocd.models.Puzzle.examples property).
    Returns a tuple of tuples representing each puzzle part and attempted/correct answers, as well as a boolean
    indicating wheher the answer is correct:
    (attempted_answer, correct_answer, answer_is_correct)
    None is used in place of the tuples where no answers are available. For example:
    (
        ("42", "40", False), # part a example 1
        ("60", "60", True),  # part a example 1
        ...
    ), None  # No examples for part b (yet?)
    """

    filter_ = "Solution to" if suppress_output else None
    ssh = StreamFilter(filter_=filter_)
    with ssh:
        all_attempts = []
        for example in examples:
            if example.extra is not None and extra_kwargs_parser is not None:
                kwargs = extra_kwargs_parser(example.extra)
            else:
                kwargs = dict()
            attempt = solver(example.input_data, **kwargs)
            all_attempts.append(attempt)

    answer_keys = ("answer_a", "answer_b")
    if not all(isinstance(attempt, tuple) and len(attempt) == len(answer_keys) for attempt in all_attempts):
        raise ValueError

    res_list = [None, None]
    for part_i, k in enumerate(answer_keys):
        part_results = []
        for i, example in enumerate(examples):
            answer = getattr(example, k)
            if answer is None:
                continue
            attempt_val = all_attempts[i][part_i]
            attempt = _coerce_val(attempt_val)
            correct = attempt == answer
            ex_res = (attempt, answer, correct)
            part_results.append(ex_res)

        if part_results:
            res_list[part_i] = part_results
        #

    res = tuple(res_list)
    return res


def check_examples(
        solver: Callable,
        year: int,
        day: int,
        suppress_output=True,
        verbose=True,
        extra_kwargs_parser=None
    ):
    """Checks whether the provided solver gives correct results for the given examples.
    Some examples come with an 'extra' attribute, denoting when something is a special case for that example.
    If extra_kwargs_parser is provided, it will be used to pass such a string into a keyword dict, which is passed
    to the solver."""

    puzzle = Puzzle(year=year, day=day)
    examples = puzzle.examples
    if verbose:
        print(f"Got {len(examples)} examples.")
    results = _evaluate_examples(
        solver=solver,
        examples=examples,
        suppress_output=suppress_output,
        extra_kwargs_parser=extra_kwargs_parser
    )

    res_a, res_b = results

    a_correct = all(correct for _, _, correct in res_a)
    b_missing = res_b is None and not puzzle.answered_a
    new_solver = solver_looks_new(solver)
    needs_refresh = all((a_correct, b_missing, new_solver))
    if needs_refresh:
        if verbose:
            print(f"Part 2 might have unlocked - refreshing puzzle cache...")
        puzzle._request_puzzle_page()
        examples = puzzle.examples
        if verbose:
            print(f"Got {len(examples)} examples.")
        results = _evaluate_examples(solver=solver, examples=examples, suppress_output=suppress_output)

    correct_list = []
    for part, res_ in enumerate(results):
        print(f"*** part {part + 1} ***")
        if res_ is None:
            cprint(f" - No example data{' (yet?)' if not puzzle.answered_a else '!'}.", color="light_grey")
            continue
        for i, (attempt, correct_answer, correct) in enumerate(res_):
            color = "green" if correct else "red"
            eq = "==" if correct else "!="
            cprint(f" - Example {i+1}: {attempt} {eq} {correct_answer}.", color=color)
            correct_list.append(correct)

    print()
    n_correct = sum(correct_list)
    p_correct = 100*n_correct/len(correct_list)

    msg = f"### Correct: {n_correct}/{len(correct_list)} ({p_correct:.0f}%) ###"
    cprint(msg, color="green" if all(correct_list) else "red")
    print()
    if not all(correct_list):
        print("Aborting since some examples fail...")
        sys.exit(0)


if __name__ == "__main__":
    pass
