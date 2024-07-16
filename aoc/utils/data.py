from aocd.cookies import get_working_tokens
from aocd.utils import get_owner
from aocd.exceptions import DeadTokenError
from aocd.models import Puzzle, User, default_user

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


if __name__ == "__main__":
    tokens = get_working_tokens()
    for token_, owner in tokens.items():
        print(token_, owner, get_owner(token_))
