#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 10:17:34 2020

@author: ahura
"""

from collections import Counter
from copy import deepcopy
from itertools import product

import numpy as np

test = '''.#.
..#
###'''

with open("input17.txt") as f:
    initstring = f.read().strip()


def parse_initstring(s, force_dims=3):
    res = np.array([list(line) for line in s.split("\n")])
    if force_dims:
        missing = [1 for _ in range(force_dims - res.ndim)]
        newdim = tuple(list(res.shape) + missing)
        res = res.reshape(newdim)

    return res


def expand(arr, add=1, default_char="."):
    newdims = tuple(d + 2*add for d in arr.shape)

    res = np.full(shape=newdims, fill_value=[default_char],dtype=str)

    for inds in np.ndindex(arr.shape):
        newinds = tuple(i + add for i in inds)
        state = arr[inds]
        res[newinds] = state

    return res


def index_is_inside_array(ind, arr):
    return all(0 <= i < lim for i, lim in zip(ind, arr.shape))


def get_neighbors(arr, inds, default_char="."):
    neighbors = []
    directions = [[-1, 0, 1] for _ in range(arr.ndim)]
    inds = np.array(inds)
    offsets = (np.array(tup) for tup in product(*directions)
               if not all(v == 0 for v in tup))

    for offset in offsets:
        neighbor_inds = tuple(inds + offset)
        if index_is_inside_array(neighbor_inds, arr):
            neighbor = arr[neighbor_inds]
        else:
            neighbor = default_char
        #
        neighbors.append(neighbor)


    return neighbors


class PocketDimension:
    def __init__(self, initial_states):
        self.states = initial_states

    def isactive(self, ind):
        return self.states[ind] == "#"

    def get_new_state(self, ind):
        neighbors = get_neighbors(self.states, ind)
        counts = Counter(neighbors)
        res = "."
        if self.isactive(ind) and counts.get("#", 0) in (2, 3):
            res = "#"
        elif not self.isactive(ind) and counts.get("#", 0) == 3:
            res = "#"
        return res

    def update_states(self):
        self.states = expand(self.states)
        newstates = deepcopy(self.states)
        for ind in np.ndindex(self.states.shape):
            newstate = self.get_new_state(ind)
            newstates[ind] = newstate

        self.states = newstates

    def cycle(self, n=1):
        for _ in range(n):
            self.update_states()


    def count_active(self):
        return sum(self.isactive(i) for i in np.ndindex(self.states.shape))

# Test
m_test = parse_initstring(test)
pd_test = PocketDimension(initial_states=m_test)
pd_test.cycle(6)
print(pd_test.count_active())
if pd_test.count_active() == 112:
    print("Success")


### Star 1
m = parse_initstring(initstring)
pd = PocketDimension(initial_states = m)
pd.cycle(6)
print(pd.count_active())

### Star 2
m2 = parse_initstring(initstring, force_dims=4)
hyperdim = PocketDimension(m2)
hyperdim.cycle(6)
print(hyperdim.count_active())