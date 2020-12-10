#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:53:01 2020

@author: ahura
"""

with open("input05.txt") as f:
    raw = [line.strip() for line in f.readlines()]

def readrow(s):
    binary = s.replace("F", "0").replace("B", "1")
    res = int(binary, 2)
    return res


def readcol(s):
    binary = s.replace("L", "0").replace("R", "1")
    res = int(binary, 2)
    return res


def parse(seat_code):
    rowcode = seat_code[:7]
    i = readrow(rowcode)
    colcode = seat_code[-3:]
    j = readcol(colcode)

    return i, j


def get_seat_id(i, j):
    res = i*8 + j
    return res


seats = [parse(seat_code) for seat_code in raw]

seat_IDs = [get_seat_id(i, j) for i,j in seats]

### Star 1
print(max(seat_IDs))

### Star 2
all_IDs = set(seat_IDs)

for i in range(128):
    for j in range(8):
        sid = get_seat_id(i, j)

        seat_is_free = sid not in all_IDs
        neighbours_exist = all(val+sid in all_IDs for val in (1, -1))
        correct = seat_is_free and neighbours_exist
        if correct:
            print(sid)