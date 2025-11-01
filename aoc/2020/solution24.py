#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:29:06 2020

@author: ahura
"""

from collections import Counter

test = '''sesenwnenenewseeswwswswwnenewsewsw
neeenesenwnwwswnenewnwwsewnenwseswesw
seswneswswsenwwnwse
nwnwneseeswswnenewneswwnewseswneseene
swweswneswnenwsewnwneneseenw
eesenwseswswnenwswnwnwsewwnwsene
sewnenenenesenwsewnenwwwse
wenwwweseeeweswwwnwwe
wsweesenenewnwwnwsenewsenwwsesesenwne
neeswseenwwswnwswswnw
nenwswwsewswnenenewsenwsenwnesesenew
enewnwewneswsewnwswenweswnenwsenwsw
sweneswneswneneenwnewenewwneswswnese
swwesenesewenwneswnwwneseswwne
enesenwswwswneneswsenwnewswseenwsese
wnwnesenesenenwwnenwsewesewsesesew
nenewswnwewswnenesenwnesewesw
eneswnwswnwsenenwnwnwwseeswneewsenese
neswnwewnwnwseenwseesewsenwsweewe
wseweeenwnesenwwwswnew'''.split("\n")

directions = ["e", "w", "se", "ne", "sw", "nw"]

def parse_line(line):
    valid = set(directions)
    res = []
    buffer = ""
    for char in line:
        buffer += char
        if buffer in valid:
            res.append(buffer)
            buffer = ""
        #
    assert buffer == ""
    return res


dir2coords = {
    "e": (1, 0),
    "w": (-1, 0),
    "ne": (0.5, 1),
    "nw": (-0.5, 1),
    "se": (0.5, -1),
    "sw": (-0.5, -1)}


def sum_vectors(*args):
    assert len(Counter(len(v) for v in args)) == 1
    res_list = [0 for _ in args[0]]
    for tup in args:
        for i, elem in enumerate(tup):
            res_list[i] += elem
        #
    res = tuple(res_list)
    return res


def resolve_path(path):
    vecs = [dir2coords[step] for step in path]
    res = sum_vectors(*vecs)
    return res


### Star 1
with open("input24.txt") as f:
    lines = [line.strip() for line in f]

# paths = [parse_line(line) for line in test]
paths = [parse_line(line) for line in lines]

tiles = [resolve_path(path) for path in paths]
tilecounts = Counter(tiles)
flipcounts = Counter(tilecounts.values())
black_tiles = [k for k, v in tilecounts.items() if v % 2 == 1]
n_black = len(black_tiles)
print(n_black)

### Star 2
def get_adjacent_tiles(tile):
    res = []
    for coord in dir2coords.values():
        neighbor = sum_vectors(tile, coord)
        res.append(neighbor)
    return res

def count_black_neighbors(tile, black_tiles):
    neighbors = get_adjacent_tiles(tile)
    n_black = sum(neighbor in black_tiles for neighbor in neighbors)
    return n_black


black_tiles = set(black_tiles)
for i in range(100):
    neighborhoods = (get_adjacent_tiles(tile) for tile in black_tiles)
    candidate_tiles = set(sum(neighborhoods, [])).union(black_tiles)
    flip_white = []
    flip_black = []
    for tile in candidate_tiles:
        n_black_neighbors = count_black_neighbors(tile, black_tiles)
        if tile in black_tiles and (n_black_neighbors == 0 or n_black_neighbors > 2):
            flip_white.append(tile)
        elif n_black_neighbors == 2:
            flip_black.append(tile)
        #
    for tile in flip_white:
        black_tiles.remove(tile)
    for tile in flip_black:
        black_tiles.add(tile)

print(len(black_tiles))
