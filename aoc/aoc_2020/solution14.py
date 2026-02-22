#  · .· ·`  *  `.· *.· ·`  •·+`    ·  * .·`.· +.  ·  ·. *·`  .·   . ·`     *·  ·
# ·.  `·*. + ·• ·  `·*. ·   *`+ ·. Docking Data • `·   ·.   ·+*·`   .+·  *·  ·`.
# `*.·  `·.`·*  `*.·   https://adventofcode.com/2020/day/14     *· ·  + ·` ·`.*·
# . ·`*·  · ` ·.  ·.  ·*`   ··   ·+  *`.  ·  • `··*  *· ` .  ·` ·.* `    ·` .*·`

import re
from itertools import product


def parse(s: str) -> list[str]:
    res = [line.strip() for line in s.splitlines()]
    return res


def binarray_from_number(num: int, n_bits=36) -> list[int]:
    """Returns the binary representation of the input number, using the
    specified number of bits, as a list, e.g.
    (num=5, n_bits=4) -> [0, 1, 0, 1]"""

    res = [0 for _ in range(n_bits)]
    running = num
    for i in reversed(range(n_bits)):
        res[i] = running & 1
        running >>= 1

    return res


def int_from_binarray(ba: list[int]) -> int:
    """Takes a binary array and computes the corresponding integer in base 10"""
    
    res = sum(digit*2**i for i, digit in enumerate(reversed(ba)))
    return res


class ReaderThingy:
    def __init__(self, initial_mem: dict[int, int]|None=None, initial_mask: str|None=None, n_bits=36) -> None:
        if not initial_mem:
            initial_mem = {}
        if initial_mask is None:
            initial_mask = "".join(n_bits*['X'])

        self.mem: dict[int, int] = initial_mem
        self.mask: str = initial_mask

    def maskdict_from_string(self, maskstring):
        res = {i: int(s) for i, s in enumerate(maskstring) if s != 'X'}
        return res

    def mask_number(self, val: int) -> int:
        binarray = binarray_from_number(val)
        resba = [val for val in binarray]
        mask = self.maskdict_from_string(self.mask)

        for i, bit in mask.items():
            resba[i] = bit
            #
        res = int_from_binarray(resba)
        
        return res

    def store_number(self, ind: int, val: int) -> None:
        masked = self.mask_number(val)
        self.mem[ind] = masked

    def parse_mem(self, line: str) -> tuple[int, int]:
        pattern = r'mem\[(\d*)\] = (\d*)'
        m = re.match(pattern, line)
        assert m
        ind = int(m.group(1))
        val = int(m.group(2))

        return ind, val

    def run_line(self, s: str) -> None:
        """Executes a single line of a program"""
        if s.startswith("mem"):
            ind, val = self.parse_mem(s)
            self.store_number(ind, val)
        elif s.startswith("mask"):
            mask = s.split("mask = ")[1]
            self.mask = mask
        #

    def read(self) -> dict[int, int]:
        res = {i: val for i, val in self.mem.items()}
        return res
    #


def unpack_floating(ba: list[int|str], x="X") -> list[list[int]]:
    """Computes all possible binary representations that may occur from substitutin
    combinations of 0 and 1 in place of 'X' in the input."""
    inds = [i for i, char in enumerate(ba) if char == x]

    results = []
    n_rands = len(inds)
    combs = product(*(n_rands*[[0, 1]]))
    for comb in combs:
        temp = [c if isinstance(c, int) else -1 for c in ba]

        for ind, newbit in zip(inds, comb):
            temp[ind] = newbit
        results.append(temp)

    return results


class ReaderThingyV2(ReaderThingy):
    def maskdict_from_string(self, maskstring):
        res = {}
        for i, s in enumerate(maskstring):
            if s == "1":
                res[i] = 1
            elif s == "X":
                res[i] = "X"

        return res

    def mask_number(self, val):
        binarray = binarray_from_number(val)
        d = self.maskdict_from_string(self.mask)
        for i, newbit in d.items():
            binarray[i] = newbit

        all_binarrays = unpack_floating(binarray)
        res = [int_from_binarray(ba) for ba in all_binarrays]

        return res

    def store_number(self, ind, val):
        masked = self.mask_number(ind)
        for address in masked:
            self.mem[address] = val
        #
    #


def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)

    reader = ReaderThingy()
    for line in program:
        reader.run_line(line)

    d = reader.read()
    star1 = sum(d.values())
    print(f"Solution to part 1: {star1}")

    r2 = ReaderThingyV2()
    for line in program:
        r2.run_line(line)

    star2 = sum(r2.read().values())
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 14
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
