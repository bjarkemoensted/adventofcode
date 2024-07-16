#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:06:33 2020

@author: ahura
"""

import re

tests = {
    "2 * 3 + (4 * 5)": 26,
    "5 + (8 * 3 + 9 + 3 * 4 * 3)": 437,
    "5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))": 12240,
    "((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2": 13632
}

def extract_paratheses(s):
    res = []
    level = 0
    parmap = {"(": 1, ")": -1}
    buffer = ""
    for char in s:
        if char in parmap:
            level += parmap[char]
            if level == 0:
                res.append(buffer+char)
                buffer = ""
            #
        if level > 0:
            buffer += char
        #
    return res


class Resolver:
    def get_components(self, expr):
        components = re.findall(r'([\w\+\*]+)', expr)
        return components

    def evaluate_components(self, components):
        res = int(components[0])
        for i in range(1, len(components), 2):
            mod = " ".join([components[i+step] for step in (0, 1)])
            hack = str(res) + " " + mod
            res = eval(hack)
        return res

    def __call__(self, expr):
        for parenthesis in extract_paratheses(expr):
            content = parenthesis[1:-1]
            evaluated = str(self.__call__(content))
            expr = expr.replace(parenthesis, evaluated)

        components = self.get_components(expr)
        res = self.evaluate_components(components)

        return res

resolve = Resolver()
if not all(resolve(expr) == answer for expr, answer in tests.items()):
    raise ValueError("Nope!")

### Star 1
with open("input18.txt") as f:
    exprs = [line.strip() for line in f]

results = [resolve(expr) for expr in exprs]
print(sum(results))

### Star 2
class AdvancedResolver(Resolver):
    def get_components(self, expr):
        return expr.split(" * ")

    def evaluate_components(self, components):
        res = 1
        for component in components:
            res *= eval(component)
        return res

resolve2 = AdvancedResolver()
results2 = [resolve2(expr) for expr in exprs]
print(sum(results2))
