#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 08:03:18 2020

@author: ahura
"""

import networkx as nx


def parseline(line):
    line = line.strip()
    inst, val = line.split()
    val = int(val)
    return inst, val

with open("input08.txt") as f:
    code = [parseline(line) for line in f]


### Star 1 (ugly, sry)
class Executor:
    def __init__(self, code, startat=0, acc=0):
        self.code = code
        self.i = startat
        self.acc = acc
        self.lines_run = set([])
        self.keeprunning = True
        self.reached_end = False

    def iterate(self):
        if self.i in self.lines_run:
            self.keeprunning = False
            return
        else:
            self.lines_run.add(self.i)

        inst, val = self.code[self.i]

        if inst == "jmp":
            self.i += val
        else:
            if inst == "acc":
                self.acc += val
            self.i += 1
        self.reached_end = self.i == len(self.code)


    def run_all(self, nmax = 10000):
        n = 0
        while self.keeprunning and n <= nmax:
            self.iterate()
            n += 1
        return self.acc


executor = Executor(code=code)
res = executor.run_all()
print(res)

### Star 2
G = nx.DiGraph()
for u, (inst, val) in enumerate(code):
    acc = val if inst == 'acc' else 0
    G.add_node(u, acc=acc, inst=inst, val=val)

    increment = val if inst == "jmp" else 1
    v = u + increment
    G.add_edge(u, v)

# Get the instructions that will eventually lead to the code terminating
endnode = len(code)
terminating_nodes = {n for n in G.nodes() if nx.has_path(G, n, endnode)}

# Search for a node that can be modified to point to a terminating node
fixed = False
current_node = 0
while not fixed:
    inst = nx.get_node_attributes(G, "inst")[current_node]
    val = nx.get_node_attributes(G, "val")[current_node]
    next_node = list(nx.neighbors(G, current_node))[0]

    # Find out where node could point to if we exchanged nop <-> jmp instr
    alternative = -1
    if inst == "nop":
        alternative = current_node + val
    elif inst == "jmp":
        alternative = current_node + 1

    # Change intruction if that would terminate the code
    if alternative in terminating_nodes:
        G.remove_edge(current_node, next_node)
        G.add_edge(current_node, alternative)
        fixed = True
    else:
        current_node = next_node
    #

# Sum the increments to acc along the path
path = nx.shortest_path(G, 0, endnode)
acc_increments = [nx.get_node_attributes(G, 'acc')[node] for node in path[:-1]]
print(sum(acc_increments))
