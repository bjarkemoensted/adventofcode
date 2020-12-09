#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 08:08:16 2020

@author: ahura
"""

with open("input9.txt") as f:
    numbers = [int(line.strip()) for line in f]


def find_first_invalid(numbers, preamble_len = 25):
    for idx in range(preamble_len, len(numbers)):
        preamble = set(numbers[idx - preamble_len: idx])
        num = numbers[idx]

        valid = False
        for a in preamble:
            b = num - a
            if b in preamble:
                valid = True
            #
        if not valid:
            return num


### Star 1
badnum = find_first_invalid(numbers, preamble_len=25)
print(badnum)


### Star 2
def find_contiguous_sum(numbers, target):
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            contiguous = numbers[i:j]
            sum_ = sum(contiguous)
            if sum_ == target:
                return contiguous
            if sum_ > target:
                break


res = find_contiguous_sum(numbers, target=badnum)
print(min(res) + max(res))