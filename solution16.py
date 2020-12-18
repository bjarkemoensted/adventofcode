#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 08:07:39 2020

@author: ahura
"""

import numpy as np

with open("input16.txt") as f:
    classstring, myticketstring, nearbyticketstring = f.read().split("\n\n")

def parse_class(s):
    field2allowed = {}
    for line in s.split("\n"):
        field, remainder = line.split(": ")
        allowed = set([])
        for substring in remainder.split(" or "):
            a, b = (int(s) for s in substring.split("-"))
            new_allowed_numbers = set(range(a, b+1))
            allowed |= new_allowed_numbers
        #
        field2allowed[field] = allowed

    return field2allowed

field2allowed = parse_class(classstring)


def parseticket(s):
    lines = s.split("\n")
    res = []
    for line in lines[1:]:
        if not line:
            continue
        newnums = [int(s) for s in line.strip().split(",")]
        res.append(newnums)
    return res

myticket = parseticket(myticketstring)[0]
nearbytickets = parseticket(nearbyticketstring)

### Star 1
all_allowed_values = set([])
for allowed in field2allowed.values():
    all_allowed_values |= allowed

wrong_numbers = []
for ticket in nearbytickets:
    for number in ticket:
        if number not in all_allowed_values:
            wrong_numbers.append(number)
        #
    #

print(sum(wrong_numbers))

### Star 2
valid_tickets = np.array([ticket for ticket in nearbytickets
                          if all(n in all_allowed_values for n in ticket)])

# Get the set of observed values for each position on the tickets
field_values = [set(vals) for vals in valid_tickets.T]

# For each index in the tickets, get list of all fields consistent w. values
possible_fields = [
    [field for field, allowed in field2allowed.items() if vals.issubset(allowed)]
    for vals in field_values]

# Sort index by number of possible fields, ascending
inds_by_options = np.argsort([len(vals) for vals in possible_fields])

# Method of exclusion. Start with ind with only one possible field
fields = [None for _ in possible_fields]
assigned = set([])
for ind in inds_by_options:
    candidates = set(possible_fields[ind]) - assigned
    if len(candidates) == 1:
        field = list(candidates)[0]
        fields[ind] = field
        assigned.add(field)
    else:
        raise ValueError('Not possible to uniquely assign fields')

departure_vals = [myticket[ind] for ind, field in enumerate(fields)
                  if field.startswith("departure")]
res = 1
for val in departure_vals:
    res *= val
print(res)