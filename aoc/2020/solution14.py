#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 07:59:50 2020

@author: ahura
"""

import re
from itertools import product


def binarray_from_number(num, n_bits=36):
    res = [int(val) for val in '{0:0b}'.format(num)]
    res = (n_bits - len(res))*[0] + res
    if len(res) > n_bits:
        raise ValueError
    return res


with open("input14.txt") as f:
    lines = [line.strip() for line in f]

def int_from_binarray(ba):
    s = "".join([str(dig) for dig in ba])
    res = int(s, 2)
    return res


### Star 1
class ReaderThingy:
    def __init__(self, initial_mem=None, initial_mask=None, n_bits=36):
        if not initial_mem:
            initial_mem = {}
        if initial_mask is None:
            initial_mask = "".join(n_bits*['X'])
        self.mem = initial_mem
        self.mask = initial_mask

    def maskdict_from_string(self, maskstring):
        res = {i: int(s) for i, s in enumerate(maskstring) if s != 'X'}
        return res

    def mask_number(self, val):
        binarray = binarray_from_number(val)
        resba = [val for val in binarray]
        mask = self.maskdict_from_string(self.mask)

        for i, bit in mask.items():
            resba[i] = bit
            #
        res = int_from_binarray(resba)
        return res

    def store_number(self, ind, val):
        masked = self.mask_number(val)
        self.mem[ind] = masked

    def parse_mem(self, line):
        pattern = r'mem\[(\d*)\] = (\d*)'
        m = re.match(pattern, line)
        ind = int(m.group(1))
        val = int(m.group(2))

        return ind, val

    def run_line(self, s):
        if s.startswith("mem"):
            ind, val = self.parse_mem(s)
            self.store_number(ind, val)
        elif s.startswith("mask"):
            mask = s.split("mask = ")[1]
            self.mask = mask
        #
    def read(self):
        res = {i: val for i, val in self.mem.items()}
        return res

reader = ReaderThingy()
for line in lines:
    reader.run_line(line)

d = reader.read()
print(sum(d.values()))

### Star 2
def unpack_floating(ba, x="X"):

    inds = [i for i, char in enumerate(ba) if char == x]

    results = []
    n_rands = len(inds)
    combs = product(*(n_rands*[[0, 1]]))
    for comb in combs:
        temp = [c for c in ba]
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


r2 = ReaderThingyV2(initial_mask=[{}])
for line in lines:

    r2.run_line(line)

d2 = r2.read()
print(sum(d2.values()))