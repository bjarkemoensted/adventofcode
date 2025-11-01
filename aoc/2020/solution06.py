#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 09:20:36 2020

@author: ahura
"""


from collections import Counter

with open("input06.txt") as f:
    chunks = f.read().split("\n\n")


def parse(chunk):
    strings = chunk.strip().split("\n")
    n_persons = len(strings)
    counts = Counter(sum([list(s) for s in strings], []))
    return n_persons, counts


### Star 1

running = 0
for chunk in chunks:
    n, counts = parse(chunk)
    running += len(counts)

print(running)

### Star 2
running2 = 0
for chunk in chunks:
    n, counts = parse(chunk)
    contribution = sum(v == n for v in counts.values())

    running2 += contribution

print(running2)