#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 08:04:47 2020

@author: ahura
"""

numbers = [12,20,0,6,1,17,7]

def run_game(stopat):
    num2lastspoken = {}
    number = None
    lastnumber = None
    for i in range(stopat):
        lastnumber = number
        if i < len(numbers):
            number = numbers[i]
        elif lastnumber not in num2lastspoken:
            number = 0
        else:
            age = i - num2lastspoken[number]
            number = age

        num2lastspoken[lastnumber] = i
    return number

### Star 1
res1 = run_game(stopat=2020)
print(res1)

###Star 2
res2 = run_game(stopat=30000000)
print(res2)




