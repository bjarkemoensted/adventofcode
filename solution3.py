#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:36:26 2020

@author: ahura
"""

import numpy as np


def scan_trajectory(map_, direction, initial=None, charmap=None):
    if not charmap:
        charmap = {"#": 1, '.': 0}
    if not initial:
        initial = (0, 0)
    i, j = initial

    right, down = direction
    assert down > 0

    height, width = map_.shape
    n_trees = charmap[map_[i, j]]

    while i < height-1:
        i = i + down
        j = (j + right) % width

        char = map_[i, j]
        n_trees += charmap[char]

    return n_trees


with open("input3.txt") as f:
    map_ = np.array([list(line.strip()) for line in f])


### Star 1

n = scan_trajectory(map_, (3, 1))
print(n)

### Star 2

paths = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]

ntrees = []
for path in paths:
    ntrees.append(scan_trajectory(map_, direction=path))

prod = 1
for x in ntrees:
    prod *= x

print(prod)