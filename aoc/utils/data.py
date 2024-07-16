from aocd.cookies import get_working_tokens
from aocd.utils import get_owner
from aocd.exceptions import DeadTokenError
from aocd.models import Puzzle, User, default_user

import contextlib
import os
from termcolor import cprint
from typing import Callable

from aoc.utils import config


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


def check_examples(solver: Callable, year:int, day: int):
    puzzle = Puzzle(year=year, day=day)
    examples = puzzle.examples
    succes = []
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            all_attempts = [solver(example.input_data) for example in examples]

    answer_keys = ("answer_a", "answer_b")
    for part in range(2):
        n_answers = len(succes)
        print(f"*** part {part + 1} ***")
        for i, example in enumerate(examples):
            k = answer_keys[part]
            answer = getattr(example, k)
            if answer is None:
                continue

            attempt = all_attempts[i][part]
            correct = answer == attempt
            color = "green" if correct else "red"
            eq = "==" if correct else "!="
            cprint(f" - Example {i+1}: {attempt} {eq} {answer}.", color=color)
            succes.append(correct)
        all_missing = len(succes) == n_answers
        if all_missing:
            cprint(" - No data yet.", color="light_grey")

    print()
    n_correct = sum(succes)
    p_correct = 100*n_correct/len(succes)

    msg = f"### Correct: {n_correct}/{len(succes)} ({p_correct:.0f}%) ###"
    cprint(msg, color="green" if all(succes) else "red")
    print()


if __name__ == "__main__":
    tokens = get_working_tokens()
    for token_, owner in tokens.items():
        print(token_, owner, get_owner(token_))
