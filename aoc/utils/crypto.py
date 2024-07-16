from nacl import pwhash, secret
from getpass import getpass
from pathlib import Path

from aoc.utils import config


def _salt():
    salt = b'\xff\xb5L6\x87\\\x88\xbf\xf4\xcaw\xfau\xda\xbd\xd5'
    return salt


def ensure_password_is_set():
    d = config.read_secrets()
    k = config.password_field
    if k not in d:
        password = getpass("Enter a secret string for encryption")
        d_up = {k: password}
        config.update_secrets(**d_up)


def _read_password():
    d = config.read_secrets()
    k = config.password_field
    s = d[k]
    res = bytes(s, encoding="utf8")
    return res


def _box_from_password(password: bytes) -> secret.SecretBox:
    kdf = pwhash.argon2i.kdf
    salt = _salt()
    key = kdf(secret.SecretBox.KEY_SIZE, password, salt=salt)
    box = secret.SecretBox(key)
    return box


class Box:
    __inst = None
    def __new__(cls):
        if cls.__inst is None:
            cls.__inst = super().__new__(cls)
        return cls.__inst

    def __init__(self):
        password = _read_password()
        self._box = _box_from_password(password=password)

    def _apply(
            self,
            operation: str,
            fn_in: str|Path,
            fn_out: str|Path,
            overwrite_if_exists=False,
            verbose=False
        ) -> None:

        fun = getattr(self._box, operation)
        if Path(fn_out).is_file() and not overwrite_if_exists:
            if verbose:
                print(f"{fn_out} already exists. Skipping.")

        with open(fn_in, "rb") as fin:
            content = fin.read()

        with open(fn_out, "wb") as fout:
            new_content = fun(content)
            fout.write(new_content)

    def encrypt(self, fn, *args, **kwargs):
        fn_out = Path(fn).with_suffix(config.ext_encrypted)
        self._apply(operation="encrypt", fn_in=fn, fn_out=fn_out, *args, **kwargs)

    def decrypt(self, fn, *args, **kwargs):
        fn_out = Path(fn).with_suffix(config.ext_data)
        self._apply(operation="decrypt", fn_in=fn, fn_out=fn_out, *args, **kwargs)


def _find_ext(ext: str):
    folder = config.root_dir

    res = []
    for day in config.days_range:
        for fn_template in (config.input_filename, config.example_filename):
            fn = fn_template.format(day=day)
            path = Path(fn).with_suffix(ext)
            for hit in folder.rglob(path):
                res.append(hit)
            #
        #

    return res


def encrypt_all():
    files = _find_ext(ext=config.ext_data)
    box = Box()
    for fn in files:
        box.encrypt(fn)
    #


def decrypt_all():
    files = _find_ext(ext=config.ext_encrypted)
    box = Box()
    for fn in files:
        box.decrypt(fn)
    #


if __name__ == '__main__':
    pass
