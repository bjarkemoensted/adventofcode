# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:18:01 2020

@author: BMM
"""

import itertools
from functools import reduce

with open("input01.txt") as f:
    nums = [int(line) for line in f.readlines()]

### star 1
s = set(nums)
for num in nums:
    if num < 2020 / 2:
        continue
    pair = 2020 - num
    if pair in s:
        print(num, pair, num*pair)
        
        
### star 2
for comb in itertools.combinations(nums, 3):
    if sum(comb) == 2020:
        print(*comb, reduce(lambda a,b: a*b, comb))