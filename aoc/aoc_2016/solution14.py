# ·.` · `` ··*   ·  *`·. `*··  ` *     ·.·     + `. · ·*    ··`*.· .  `*· ·. •·`
# •·  `·.*·`   ··*.`        *· ·   One-Time Pad +· ·*  .·`     `·*.· .··* `* · .
# ` ·`* · `  ·  ` ·*   https://adventofcode.com/2016/day/14   · *.·  ` .` *·· `·
# ·.*··` ·+.  · *`  ·· *    ` *  ·`.   ·     ·`  · *.  ` ·+· ` ·`   · *  · ·` .*


from functools import cache
import hashlib


def parse(s: str):
    res = s
    return res


def get_n_repeated_chars(s: str, n: int):
    """Given a string and integer, returns a list of characters repeated n times in the string, in the order they occur.
    For instance ('abbccaabb', 2) -> ['b', 'c', 'a', 'b']."""
    res = []

    for i, char in enumerate(s[:-n + 1]):
        if all(c == char for c in s[i:i + n]):
            res.append(char)

    return res


class KeyGen:
    def __init__(self, salt: str, n_rehashes=0, verbose=False):
        self.salt = salt
        self.n_rehashes = n_rehashes
        self.verbose = verbose

        self.n_repeats_short = 3
        self.n_repeats_long = 5
        self.window = 1000

    @cache
    def scramble(self, input_):
        """Computes the hash of a given input"""
        s = self.salt + str(input_)
        for _ in range(1 + self.n_rehashes):
            s = hashlib.md5(s.encode("utf-8")).hexdigest()

        return s

    def get_short(self, ind: int):
        """Takes and index and returns the first 'short repetition', e.g. '777' in the hash og that index.
        Returns None if none exist."""

        hash_ = self.scramble(ind)
        repeats = get_n_repeated_chars(hash_, n=self.n_repeats_short)
        res = repeats[0] if repeats else None
        return res

    @cache
    def get_longs(self, ind: int):
        """Takes an index and returns a set of 'long repititions', e.g. '77777' in the hash of that index.
        Returns an empty set if no such repititions exist in the string."""
        hash_ = self.scramble(ind)
        repeats = get_n_repeated_chars(hash_, n=self.n_repeats_long)
        return set(repeats)

    def is_key(self, ind: int):
        """Determines whether the index is a key"""
        char = self.get_short(ind)
        if char is None:
            return False

        for i in range(ind+1, ind+self.window+1):
            if char in self.get_longs(i):
                if self.verbose:
                    print(f"Key at index {ind}: Repeated char: '{char}' found at index {i}.")
                return True

        return False


def get_first_n_keys(keygen: KeyGen, n: int) -> list[int]:
    keys: list[int] = []
    ind = 0
    while len(keys) < n:
        if keygen.is_key(ind):
            keys.append(ind)
        ind += 1
        if ind % 1000 == 0:
            print(f"Reached index {ind} - found {len(keys)} keys.", end="\r")
        #
    print()

    return keys


def solve(data: str) -> tuple[int|str, int|str]:
    salt = parse(data)

    keygen = KeyGen(salt=salt)
    n = 64
    keys = get_first_n_keys(keygen=keygen, n=n)
    star1 = keys[-1]
    print(f"Key number {n} is produced by index {star1}.")

    n_rehashes = 2016
    stretch_keygen = KeyGen(salt=salt, n_rehashes=n_rehashes)
    keys2 = get_first_n_keys(keygen=stretch_keygen, n=n)
    star2 = keys2[-1]
    print(f"Key number {n} with stretched hashing is produced by index {star2}.")

    return star1, star2


def main() -> None:
    year, day = 2016, 14
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
