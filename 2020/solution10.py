#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 08:05:11 2020

@author: ahura
"""

from collections import Counter
from itertools import chain, combinations

with open("input10.txt") as f:
    ratings = [int(line) for line in f]


### Star 1
ordered = sorted(ratings)
ordered = [0]+ordered+[ordered[-1]+3]
diffs = [ordered[i+1] - ordered[i] for i in range(len(ordered) - 1)]
counts = Counter(diffs)

solution = counts[1]*counts[3]
print(solution)

### Star 2
def chop(arr, dist_cut=3):
    buffer = [arr[0]]
    for elem in arr[1:]:
        if elem - buffer[-1] == dist_cut:
            yield buffer
            buffer = []
        buffer.append(elem)
        #
    if buffer:
        yield buffer


def make_subsets(arr):
    combs = (combinations(arr, i) for i in range(len(arr) + 1))
    subs = chain.from_iterable(combs)
    return subs


def brute_force(chunk):
    if len(chunk) <= 2:
        return 1

    middle = chunk[1:-1]
    subsets = make_subsets(middle)
    res = 0
    for subset in subsets:
        temp = [chunk[0]] + list(subset) + [chunk[-1]]
        valid = all(temp[i+1] - temp[i] <= 3 for i in range(len(temp) - 1))
        res += valid
    return res


def count_permutations(ratings):
    res = 1
    cache = {}
    chunks = chop(ratings)
    for chunk in chunks:
        n_components = len(chunk)
        try:
            res *= cache[n_components]
        except KeyError:
            perm = brute_force(chunk)
            cache[n_components] = perm
            res *= perm
        #
    return res

N = count_permutations(ordered)
print(N)
