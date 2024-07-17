from aocd.cookies import get_working_tokens
from aocd.utils import get_owner
from aocd.exceptions import DeadTokenError
from aocd.models import Puzzle, User, default_user

import contextlib
import dis
from hashlib import md5
import os
import sys
from termcolor import cprint
from typing import Callable

from aoc.utils.temp import TempCache


def token_works(token: str) -> bool:
    try:
        _ = get_owner(token)
        return True
    except DeadTokenError:
        return False


def read_data(year: int, day: int, token=None):
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


def _evaluate_examples(solver: Callable, examples: list, suppress_output: bool):
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

    with open(os.devnull, 'w') as devnull:
        # Redirect to devnull if we're going quiet
        newout = devnull if suppress_output else sys.stdout
        with contextlib.redirect_stdout(newout):
            all_attempts = [solver(example.input_data) for example in examples]
        #

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


def check_examples(solver: Callable, year: int, day: int, suppress_output=True, verbose=True):
    puzzle = Puzzle(year=year, day=day)
    examples = puzzle.examples
    results = _evaluate_examples(solver=solver, examples=examples, suppress_output=suppress_output)

    res_a, res_b = results

    a_correct = all(correct for _, _, correct in res_a)
    b_missing = res_b is None
    new_solver = solver_looks_new(solver)
    needs_refresh = all((a_correct, b_missing, new_solver))
    if needs_refresh:
        if verbose:
            print(f"Part 2 might have unlocked - refreshing puzzle cache...")
        puzzle._request_puzzle_page()
        examples = puzzle.examples
        results = _evaluate_examples(solver=solver, examples=examples, suppress_output=suppress_output)

    correct_list = []
    for part, res_ in enumerate(results):
        print(f"*** part {part + 1} ***")
        if res_ is None:
            cprint(" - No data yet.", color="light_grey")
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
    tokens = get_working_tokens()
    for token_, owner in tokens.items():
        print(token_, owner, get_owner(token_))
