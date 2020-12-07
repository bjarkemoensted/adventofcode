#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 08:17:22 2020

@author: ahura
"""

import networkx as nx
import re

with open("input7.txt") as f:
    lines = f.readlines()

pattern = "(\d+) ([\w\s]+) bag"
def parse(line):
    parent, children = line.strip().split("bags contain")
    parent = parent.strip()
    childattrs = []

    for nstring, color in re.findall(pattern=pattern, string=children):
        n = int(nstring)
        childattrs.append((n, color.strip()))

    return parent, childattrs


G = nx.DiGraph()
for line in lines:
    u, tups = parse(line)
    for n, v in tups:

        G.add_edge(u, v, n_contains=n)


### Star 1
all_colors = set(G.nodes())
n_paths = 0
goal = "shiny gold"
for start in (all_colors - set([goal])):
    n_paths += nx.has_path(G, start, goal)
    # dist = nx.shortest_path_length(G, start, goal)
    # print(dist)

print(n_paths)


### Star 2
def scan_recursive(graph, start_node):
    return 1 + sum(graph[start_node][v]["n_contains"]*scan_recursive(graph, v)
                   for v in graph.neighbors(start_node))


res = scan_recursive(graph=G, start_node="shiny gold")
print(res - 1)