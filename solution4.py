# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:25:43 2020

@author: BMM
"""

import re

required_fields = {"byr", "iyr", "eyr", "hgt", "hcl", "ecl", "pid"}
optional_fields = set(["cid"])


def validate_height(s):
    if not re.match("\d{2,3}\w{2}", s):
        return False
    val = int(s[:-2])
    unit = s[-2:]
    if unit == "cm":
        return 150 <= val <= 193
    elif unit == "in":
        return 59 <= val <= 76
    else:
        return False
    #


rules = {
        "byr": lambda s: bool(re.match("\d{4}", s)) and 1920 <= int(s) <= 2002,
        "iyr": lambda s: bool(re.match("\d{4}", s)) and 2010 <= int(s) <= 2020,
        "eyr": lambda s: bool(re.match("\d{4}", s)) and 2020 <= int(s) <= 2030,
        "hgt": validate_height,
        "hcl": lambda s: bool(re.match("#[0-9a-f]{6}", s)),
        "ecl": lambda s: s in "amb blu brn gry grn hzl oth".split(),
        "pid": lambda s: bool(re.match("^\d{9}$", s))
}


def parse_batch(raw):
    entries = raw.split("\n\n")
    res = []
    for entry in entries:
        d = {}
        for part in entry.split():
            field, value = part.split(":")
            d[field] = value
            #
        res.append(d)
    return res


def isvalid(entry, rules=None):
    if not all(field in entry for field in required_fields):
        return False
    if rules:
        return all(validator(entry[field]) for field, validator in rules.items())        
    else:
        return True


with open("input4.txt") as f:
    raw = f.read()
    contents = parse_batch(raw)


### Star 1
print(sum(isvalid(entry) for entry in contents))

### star 2
print(sum(isvalid(entry, rules) for entry in contents))
