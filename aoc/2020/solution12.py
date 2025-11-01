#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:26:19 2020

@author: ahura
"""

import numpy as np


def parse(s):
    s = s.strip()
    instruction = s[0]
    val = int(s[1:])
    return instruction, val

with open("input12.txt") as f:
    instructions = [parse(line) for line in f]

def rotate(vec, angle):
    theta = np.deg2rad(angle)
    M = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    new = M.dot(vec)
    res = np.rint(new).astype(int)

    return res

### Star 1
compass = {
    "N": np.array([0, 1]),
    "S": np.array([0, -1]),
    "E": np.array([1, 0]),
    "W": np.array([-1, 0])}

turns = {
    "L": +1,
    "R": -1}

class Ferry:
    def __init__(self, position, direction):
        self.position = np.array(position)
        self.direction = np.array(direction)

    def run_instruction(self, inst):
        action, val = inst
        if action in compass:
            self.position += compass[action]*val
        elif action == "F":
            self.position += self.direction*val
        elif action in turns:
            angle = turns[action]*val
            new_direction = rotate(self.direction, angle=angle)
            self.direction = new_direction

direction = [1, 0]
position = [0,0]
ferry = Ferry(position=position, direction=direction)

for instruction in instructions:
    ferry.run_instruction(instruction)

manhatten_dist = sum(abs(val) for val in ferry.position.flat)
print(manhatten_dist)

### Star 2

class WeirdFerry:
    def __init__(self, position, waypoint):
        self.position = np.array(position)
        self.waypoint = np.array(waypoint)

    def run_instruction(self, inst):
        action, val = inst
        if action in compass:
            self.waypoint += compass[action]*val
        elif action == "F":
            self.position += self.waypoint*val
        elif action in turns:
            angle = turns[action]*val
            new_waypoint = rotate(self.waypoint, angle=angle)
            self.waypoint = new_waypoint
        #
    #

wf = WeirdFerry(position=position, waypoint=[10, 1])

for instruction in instructions:
    wf.run_instruction(instruction)

res = sum(abs(val) for val in wf.position.flat)
print(res)