from ast import literal_eval as LE
from copy import deepcopy
import math

with open("input18.txt") as f:
    raw = f.read()


def parse(s):
    res = [LE(line.strip()) for line in s.split("\n")]
    return res


nums = parse(raw)


def _access_nested(arr, inds):
    ptr = arr
    for i in inds:
        ptr = ptr[i]
    return ptr


def _set_nested(arr, inds, val):
    ptr = arr
    for i in inds[:-1]:
        ptr = ptr[i]
    ptr[inds[-1]] = val


class Number:
    def __init__(self, arr, verbose=True):
        self.arr = arr
        self.verbose = verbose

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __getitem__(self, inds):
        return _access_nested(self.arr, inds)

    def __setitem__(self, inds, value):
        val = deepcopy(value)
        _set_nested(self.arr, inds, val)

    def __str__(self):
        return str(self.arr)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        arr = [self.arr, other.arr]
        res = type(self)(arr=arr)
        res.print(f"After addition: {res}.")
        res.reduce()
        return res

    def ispair(self, inds):
        return isinstance(self[inds], list) and all(isinstance(val, int) for val in self[inds])

    def _it(self, inds=None):
        if inds is None:
            inds = []
        inds = deepcopy(inds)
        for i in range(len(self[inds])):
            inds_current_level = deepcopy(inds) + [i]
            yield inds_current_level
            if isinstance(self[inds_current_level], list):
                for subinds in self._it(inds=inds_current_level):
                    yield subinds
                #
            #
        #

    def split(self, inds):
        self.print(f"Splitting at: {inds}, value={self[inds]}.")
        mid = self[inds] / 2
        a, b = math.floor(mid), math.ceil(mid)
        self[inds] = [a, b]

    def try_split(self):
        for inds in self._it():
            if isinstance(self[inds], int) and self[inds] >= 10:
                self.split(inds)
                return True
            #
        return False

    def _scan_for_number(self, inds, right=True):
        """Scans for the next number. Right and up if rigth = true, otherwise left and up"""
        inds = deepcopy(inds)
        while inds:
            if right:
                inds[-1] += 1
            else:
                inds[-1] -= 1
            falloff = inds[-1] < 0 or inds[-1] >= len(self[inds[:-1]])
            # TODO problemet er her, vi skal ogsaa "ned" i lister mode hoejre, f.eks.
            if falloff:
                inds.pop()
                continue
            else:
                if isinstance(self[inds], int):
                    return inds

    def explode(self, inds):
        self.print(f"Exploding at: {inds}, value={self[inds]}.")
        # Make sure we're exploding ints only
        a, b = self[inds]
        assert all(isinstance(val, int) for val in (a, b))
        assert len(inds) == 4

        # Replace the 'exploded' pair with a zero
        self[inds] = 0
        # Increment the nearest neighbors with a and b
        for num, right in zip((a, b), (False, True)):
            numinds = self._scan_for_number(inds, right)
            if numinds is not None:
                self[numinds] += num
            #
        #

    def try_explode(self):
        #self.print(f"Attempting explode")
        for inds in self._it():
            if self.ispair(inds) and len(inds) >= 4:
                self.explode(inds)
                return True
            #
        return False

    def reduce(self):
        done = False
        while not done:
            exploded = self.try_explode()
            if exploded:
                self.print(f"After explode: {self}")
                continue
            else:
                split = self.try_split()
                if split:
                    self.print(f"After split: {self}.")
                    continue
                #
            done = not exploded and not split


a = Number([[[[4,3],4],4],[7,[[8,4],9]]])
b = Number([1,1])

c = a + b
print(c)