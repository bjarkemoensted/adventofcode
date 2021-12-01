#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:17:16 2020

@author: ahura
"""

from copy import deepcopy
from itertools import product
import re


def parse(s):
    rulestrings, messages = [part.strip().split("\n") for part in s.split("\n\n")]
    rules = {}
    for line in rulestrings:
        rulenumstring, messstring = line.strip().split(": ")
        rulenumber = int(rulenumstring)
        rules[rulenumber] = messstring

    return rules, messages


def build_rules(d):
    res = {}
    for n, s in d.items():
        val = None
        if s.startswith('"') and s.endswith('"'):
            val = s[1]
        else:
            val = []
            for rulestring in s.split(" | "):
                theserules = [int(val) for val in rulestring.split()]
                val.append(theserules)
            #
        res[n] = val
    return res


class RuleTree:
    def __init__(self, rules):
        self.rules = deepcopy(rules)

    def traverse(self, node, depth=0):
        content = self.rules[node]
        if isinstance(content, str):
            yield content
        else:
            rules = content
            for branch in rules:
                subresults = (self.traverse(n) for n in branch)
                for tup in product(*subresults):
                    yield "".join(tup)


###Star 1
with open("input19.txt") as f:
    real_input = f.read()

rulestuff, messages = parse(real_input)
rules = build_rules(rulestuff)
ruletree = RuleTree(rules)
unmatches_messages = set(messages)

n_msg = len(messages)
for elem in ruletree.traverse(0):
    if elem in unmatches_messages:
        unmatches_messages.remove(elem)
    #

n_valid = n_msg - len(unmatches_messages)
print(n_valid)

### Star 2
def resolve_rule(ruledict, node):
    rule = ruledict[node]
    # If we hit and end node, we're done
    if isinstance(rule, str):
        return rule

    override = (8, 11)
    if node in override:
        p42 = '(?:%s)+' % resolve_rule(ruledict, 42)
        if node == 8:
            # return '(?:%s)+' % resolve_rule(ruledict, 42)
            return p42
        elif node == 11:
            p31 = resolve_rule(ruledict, 31)
            return '(?:%s)' % '|'.join(map(lambda c: p42 * c + p31 * c, range(1, 11)))

    patterns = []
    for branch in rule:
        pattern = "".join(resolve_rule(ruledict, r) for r in branch)
        patterns.append(pattern)

    res = '(?:%s)' % "|".join(patterns)
    return res


pattern = resolve_rule(rules, node=0)
regex = re.compile(pattern)

n_matches = sum(1 for w in messages if regex.fullmatch(w))
print(n_matches)
