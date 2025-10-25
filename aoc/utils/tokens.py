from aocd.cookies import get_working_tokens
from aocd.utils import get_owner
from aocd.exceptions import DeadTokenError
from aocd.models import AOCD_CONFIG_DIR
import functools
import json
from termcolor import cprint

from aoc.utils.utils import nolog

_tokens_file = AOCD_CONFIG_DIR / "tokens.json"
_token = AOCD_CONFIG_DIR / "token"
_token2id = AOCD_CONFIG_DIR / "token2id.json"


def token_works(token: str) -> bool:
    try:
        _ = get_owner(token)
        return True
    except DeadTokenError:
        return False


def read_tokens() -> dict[str, str]:
    try:
        with open(_tokens_file, "r") as f:
            d = json.load(f)
        #
    except FileNotFoundError:
        d = dict()
    
    # Empty strings seem to cause aocd to just reuse another key
    res = dict()
    for k, v in d.items():
        if v:
            res[k] = v
        else:
            cprint(f"*** Missing access token for user: {k} ***", color="red")
    
    return res


def add_tokens_from_current_session() -> None:
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
    #


def save(user2token, default_user=None) -> None:
    # Can't have nulls because aocd attempts some string ops
    user2token = {k: "" if v is None else v for k, v in user2token.items()}
    if default_user:
        default_token = user2token[default_user]
        txt = f"{default_token} <- {default_user}"
        _token.write_text(txt, encoding="utf-8")
    
    dumper = functools.partial(json.dump, indent=4, sort_keys=True)

    with open(_tokens_file, "w") as f:
        dumper(obj=user2token, fp=f)

    with open(_token2id, "w") as f:
        d = {token: user for user, token in user2token.items() if token is not None}
        dumper(obj=d, fp=f)


def fix_tokens():
    all_tokens = set([])
    all_users = set([])
    browser_tokens = []
    default_token = None

    # Browser
    browser = get_working_tokens()
    for token, _ in browser.items():
        all_tokens.add(token)
        browser_tokens.append(token)

    # user2token
    user2token_local = json.loads(_tokens_file.read_text(encoding="utf-8"))
    for user, token in user2token_local.items():
        if token is not None:
            all_tokens.add(token)
        all_users.add(user)

    # token2user
    if _token2id.is_file():
        token2user_local = json.loads(_token2id.read_text(encoding="utf-8"))
        for token, user in token2user_local.items():
            if token is not None:
                all_tokens.add(token)
            all_users.add(user)
    
    default_token = None
    txt = _token.read_text(encoding="utf-8").strip()
    if txt:
        token = txt.split()[0]
        all_tokens.add(token)
        default_token = token
    
    user2token = dict()
    default_user = None
    with nolog():
        for token in all_tokens:
            try:
                user = get_owner(token=token)
                user2token[user] = token
                if token == default_token:
                    default_user = user
            except DeadTokenError:
                pass
        #
    
    for user in all_users:
        if user not in user2token:
            user2token[user] = None
        #
    
    users = sorted(user2token.keys(), key=lambda t: (int(user2token[user] is None), user))
    user2oneind = {user: i+1 for i, user in enumerate(users)}
    oneind2user = {i: u for u, i in user2oneind.items()}

    missing_users = [user for user in users if user2token[user] is None]
    
    default_sym = "*"
    browser_sym = "B"

    def display():
        for user in users:
            i = user2oneind[user]
            i_string = str(i)
            token = user2token[user]

            ds = default_sym if user == default_user else len(default_sym)*" "
            bs = browser_sym if token in browser_tokens else len(browser_sym)*" "
            
            parts = (ds, i_string, bs, user)
            line = f"{ds}{i_string} {bs} {user}"
            line = " ".join(parts)
            color = "red" if token is None else "green"
            cprint(line, color=color)
        #

    print(f"Found {len(user2token)} user tokens - {sum(v is None for v in user2token.values())} are dead.")

    print(f"An overview of the tokens is presented below")
    print(f"The current default user is indicated with a '{default_sym}'.")
    print(f"Any tokens found in the browser storage is indicated with '{browser_sym}'.")
    print(f"Working tokens are shown in ", end="")
    cprint("green", color="green", end="")
    print(", and the missing ones in ", end="")
    cprint("red", color="red", end=".\n")
    print(f"Default user id: {default_user}.")

    display()
    print()

    new_default = None
    allowed = {i for i, user in oneind2user.items() if user not in missing_users}
    while new_default is None and len(allowed) > 0:
        raw_= input(f"Select an option in {', '.join(map(str, allowed))} (enter to keep current): ")
        try:
            ind = int(raw_.strip())
            if ind in allowed:
                new_default = oneind2user[ind]
            #
        except:
            if not raw_:
                new_default = default_user
            #
        #

    save(user2token=user2token, default_user=new_default)


if __name__ == "__main__":
    fix_tokens()
