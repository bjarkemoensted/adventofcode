#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:33:42 2020

@author: ahura
"""

import re

pattern = r"(\d*?)\-(\d*?) (\w): (.*)"

n_valid = 0
n_valid_new = 0

def isvalid_old(n_min, n_max, letter, password):
    valid = n_min <= password.count(letter) <= n_max
    return valid

def isvalid_new(pos1, pos2, letter, password):
    valid = (password[pos1-1] == letter) ^ (password[pos2-1] == letter)
    return valid

with open("input2.txt") as f:
    for line in f:
        matches = re.findall(pattern, line)
        assert len(matches) == 1
        m = matches[0]
        a, b = (int(s) for s in m[:2])
        let = m[2]
        pw = m[3]

        n_valid += isvalid_old(a, b, let, pw)

        n_valid_new += isvalid_new(a, b, let, pw)

print(n_valid)
print(n_valid_new)