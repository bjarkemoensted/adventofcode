from aocd.cookies import get_working_tokens
from aocd.utils import get_owner
from aocd.exceptions import DeadTokenError
from aocd.models import AOCD_CONFIG_DIR

import json

_tokens_file = AOCD_CONFIG_DIR / "tokens.json"


def token_works(token: str) -> bool:
    try:
        _ = get_owner(token)
        return True
    except DeadTokenError:
        return False


def read_tokens():
    try:
        with open(_tokens_file, "r") as f:
            d = json.load(f)
        #
    except FileNotFoundError:
        d = dict()
    return d


def add_tokens_from_current_session():
    """Checks for AoC tokens and updates"""
    token2owner = get_working_tokens()

    owner2token = read_tokens()
    owners_old = set(owner2token.keys())
    n_new = 0
    for token, owner in token2owner.items():
        if owner not in owners_old:
            owner2token[owner] = token
            n_new += 1
        #

    if n_new:
        with open(_tokens_file, "w") as f:
            json.dump(owner2token, f, indent=4)
        print(f"Added {n_new} token{'s' if n_new != 1 else ''} to {_tokens_file}.")
    else:
        print("No new tokens discovered.")


if __name__ == "__main__":
    add_tokens_from_current_session()
