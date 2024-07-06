import json
from nacl import pwhash, secret
import os
from pathlib import Path


def _salt():
    salt = b'\xff\xb5L6\x87\\\x88\xbf\xf4\xcaw\xfau\xda\xbd\xd5'
    return salt


_ext_code = ".py"
_ext_data = ".txt"
_ext_encrypted = ".dat"
_input_file_pattern = "input*"
_here = os.path.dirname(os.path.abspath(__file__))
_secret_file = os.path.join(_here, "secrets.json")


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


if __name__ == '__main__':
    decrypt()
