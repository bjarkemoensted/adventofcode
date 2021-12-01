#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 10:04:24 2020

@author: ahura
"""

door_public = 7573546
card_public = 17786549


def step(val, subject_number):
    res = val * subject_number
    res = res % 20201227
    return res


def get_loop_size(target_value, subject_number=7):
    running = 1
    loop = 0
    while running != target_value:
        running = step(running, subject_number)
        loop += 1
    return loop


def transform(subject_number, loop_size):
    res = 1
    for _ in range(loop_size):
        res = step(res, subject_number)
    return res

### Star 1
door_loop_size = get_loop_size(target_value=door_public)
card_loop_size = get_loop_size(target_value=card_public)

encryption_key = transform(door_public, card_loop_size)
print(encryption_key)