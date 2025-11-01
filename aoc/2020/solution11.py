#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 08:16:55 2020

@author: ahura
"""

from copy import deepcopy

import numpy as np

with open("input11.txt") as f:
    map_ = np.array([list(s.strip()) for s in f])

def get_vicinity(map_, i, j):
    res = []
    nr, nc = map_.shape
    for ii in range(max(i-1, 0), min(i+2, nr)):
        for jj in range(max(j-1, 0), min(j+2, nc)):
            if ii == i and jj == j:
                continue
            res.append(map_[ii, jj])
        #
    return res

def get_new_seat_state(seat, vicinity, crowd=4):
    seat_is_empty = seat == "L"
    vicinity_is_free = all(s != "#" for s in vicinity)
    if seat_is_empty and vicinity_is_free:
        return "#"

    seat_is_occupied = seat == "#"
    vicinity_is_crowded = sum(s == "#" for s in vicinity) >= crowd
    if seat_is_occupied and vicinity_is_crowded:
        return "L"

    return seat

### Star 1
class Simulation():
    def __init__(self, map_, crowd, get_vicinity):
        self.map_ = deepcopy(map_)
        self.stable = False
        self.crowd = crowd
        self.get_vicinity = get_vicinity

    def tick(self):
        old_state = deepcopy(self.map_)
        nr, nc = self.map_.shape
        something_changed = False
        for i in range(nr):
            for j in range(nc):
                seat_current = old_state[i, j]
                vicinity = self.get_vicinity(old_state, i, j)
                seat_new = get_new_seat_state(seat_current, vicinity, crowd=self.crowd)
                if seat_current != seat_new:
                    something_changed = True
                self.map_[i, j] = seat_new
            #
        if not something_changed:
            self.stable = True
        #

    def run(self, max_steps=1000):
        n = 0
        while not self.stable and n < max_steps:
            self.tick()
            n += 1
        if not self.stable:
            print("*** Did not converge!")
        #

    def count_occupied(self):
        n_occupied = sum(val == "#" for val in self.map_.flat)
        return n_occupied

sim = Simulation(map_, crowd=4, get_vicinity=get_vicinity)
sim.run()
n_occupied = sim.count_occupied()
print(n_occupied)

### Star 2
def line_of_sight_vicinity(map_, i, j):
    directions = []
    for a in range(-1, 2):
        for b in range(-1, 2):
            if not a == b == 0:
                directions.append((a, b))
            #
        #

    res = []
    for a, b in directions:
        ii, jj = i, j
        scan = True
        while scan:
            ii += a
            jj += b
            falloff = not all(0 <= ind < lim for ind, lim in zip((ii, jj), map_.shape))
            if falloff:
                break
            seat = map_[ii, jj]
            if seat in ("#", "L"):
                res.append(seat)
                scan = False
            #
        #
    return res

newsim = Simulation(map_, crowd=5, get_vicinity=line_of_sight_vicinity)
newsim.run()
n_new = newsim.count_occupied()
print(n_new)