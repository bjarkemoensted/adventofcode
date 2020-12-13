#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 10:22:35 2020

@author: ahura
"""

with open("input13.txt") as f:
    lines = f.readlines()
timestamp = int(lines[0])
busIDs = [int(s) for s in lines[1].split(",") if s != "x"]


### Star 1
def get_waittime(timestamp, busID):
    if timestamp % busID == 0:
        return 0
    else:
        return busID - timestamp % busID

best_route = min(busIDs, key=lambda bid: get_waittime(timestamp, bid))
waittime = get_waittime(timestamp, best_route)
print(best_route*waittime)

### Star 2
def parse_offsets(s):
    return [(int(bid), offset) for offset, bid in enumerate(s.split(",")) if bid!="x"]

idtups = parse_offsets(lines[1])

def multiply_factors(factor2multiplicity):
    res = 1
    for fac, mul in factor2multiplicity.items():
        res *= fac**mul
    return res


def prime_factor(n):
    res = {}
    running = n
    for i in range(2, n + 1):
        while running % i == 0:
            res[i] = res.get(i, 0) + 1
            running = running / i
        #
    check = multiply_factors(res)
    if check != n:
        raise ValueError("Got %d instead of %d." % (check, n))
    return res


def get_least_common_divisor(numbers):
    res = {}
    primefacs = [prime_factor(n) for n in numbers]

    for d in primefacs:
        for fac, mul in d.items():
            res[fac] = max(mul, res.get(fac, 0))
        #

    final = multiply_factors(res)
    return final


# Ugly but I don't know modular arithmetic...
def iterative_scan(idtups):
    increment = 1
    running = 1
    factors = []

    for factor, remainder in idtups:
        while get_waittime(running, factor) != remainder%factor:
            running += increment
        factors.append(factor)
        increment = get_least_common_divisor(factors)

    return running

timestamp = iterative_scan(idtups)
print(timestamp)
