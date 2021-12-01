#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:29:50 2020

@author: ahura
"""

from collections import Counter, defaultdict
from copy import deepcopy
import numpy as np

def parse(s):
    chunks = s.strip().split("\n\n")
    res = {}
    for chunk in chunks:
        lines = [line.strip() for line in chunk.split("\n")]
        tile_no = int(lines[0].split("Tile ")[1][:-1])
        arr = np.array([list(s) for s in lines[1:]])
        res[tile_no] = arr
    return res

with open("input20.txt") as f:
    tile_no2arr = parse(f.read())


def rotflip(arr):
    """Generates all rotations and flips of given input array."""
    for m in (arr, arr.T):
        for k in range(4):
            res = np.rot90(m, k=k)
            yield res
        #
    #


def get_borders(arr):
    """Returns a dictionary with the 4 borders of input array."""
    raw = {}
    raw["top"]    = arr[0,:]
    raw["bottom"] = arr[-1,:]
    raw["left"]   = arr[:, 0]
    raw["right"]  = arr[:, -1]
    res = {k: "".join(v) for k, v in raw.items()}
    return res


def remove_border(arr):
    """Removes the first and last rows and columns from input array"""
    inner = arr[1:-1, 1:-1]
    res = deepcopy(inner)
    return res


def get_possible_borders(arr):
    """Gives all the borders from all rotations/flips of input arrays.
    This means a set of all borders the 'piece' can possible interface with.
    """
    res = set([])
    for m in rotflip(arr):
        s = "".join(m[0,:])
        res.add(s)
    return res

tile2borders = {n: get_possible_borders(m) for n, m in tile_no2arr.items()}

def count_possible_matches(tile_borders, bordercounts):
    """For input set of borders, along with a countmap of all borders,
    returns the number of possible neighbors the given piece can have."""
    n_matches = sum(bordercounts.get(e, 0) > 1 for e in tile_borders)
    return n_matches

test = list(tile_no2arr.values())[2]

# Map of each border to the number of neighbors it may fir with
bordercounts = Counter(sum(map(list, tile2borders.values()), []))

tile2n_neighbors = {tile: count_possible_matches(borders, bordercounts)
                    for tile, borders in tile2borders.items()}

class Jigsaw:
    def __init__(self, shape):
        self.pieces = np.full(shape=shape, fill_value=None)

    def place(self, piece, ind):
        i, j = ind
        borders = get_borders(piece)
        match_left = j == 0 or borders["left"] == get_borders(self.pieces[i, j-1])["right"]
        match_up = i == 0 or borders["top"] == get_borders(self.pieces[i-1, j])["bottom"]
        if match_left and match_up:
            self.pieces[i, j] = piece
            return True
        else:
            return False
        #
    def merge(self):
        nrp, ncp = self.pieces.shape
        nrt, nct = [val - 2 for val in self.pieces[0,0].shape]
        res = np.full(shape=(nrp*nrt, ncp*nct), fill_value=".")

        for i, j in np.ndindex(self.pieces.shape):
            tile = self.pieces[i, j]
            img = remove_border(tile)
            nr, nc = img.shape
            for ii, jj in np.ndindex(img.shape):
                rowind = ii + nr*i
                colind = jj + nc*j
                res[rowind, colind] = img[ii, jj]
            #
        return res
    #

len_ = int(len(tile_no2arr)**.5)
remaining = {n: deepcopy(arr) for n, arr in tile_no2arr.items()}

puzzle = Jigsaw(shape=(len_, len_))

# Pick a puzzle piece to go in the top left corner
topleft_n = sorted(tile for tile, n in tile2n_neighbors.items() if n == 4)[0]
topleft = remaining[topleft_n]
for alt in rotflip(topleft):
    borders = get_borders(alt)
    matchcounts = {direction: bordercounts[border] for direction, border in borders.items()}
    if matchcounts["top"] == matchcounts["left"] == 1:
        topleft = alt
        break
    #

# Place it
overview = np.full(shape=(len_, len_), fill_value=-1)
puzzle.place(piece=topleft, ind=(0,0))
overview[0, 0] = topleft_n
del remaining[topleft_n]

def iterate_pieces(d):
    for n, arr in d.items():
        for alt in rotflip(arr):
            yield n, alt

# Place the remaining pieces
for i in range(len_):
    for j in range(len_):
        if i == j == 0:
            continue
        matches = []
        for n, arr in iterate_pieces(remaining):
            piece_fits = puzzle.place(piece=arr, ind=(i, j))
            if piece_fits:
                overview[i, j] = n
                break
            #
        del remaining[n]
    #

### Star 1
prod = 1
for i in (0, -1):
    for j in (0, -1):
        prod *= overview[i, j]

print(prod)

### Star 2
# Combine the tiles into the final image
final = puzzle.merge()

# The monster to look for
monster_string = '''------------------#-
#----##----##----###
-#--#--#--#--#--#---'''
monster = np.array([list(s) for s in monster_string.split("\n")])


def count_monsters(img, monster):
    key_inds = list(zip(*np.where(monster=="#")))
    height, width = monster.shape
    res = 0
    for i, j in np.ndindex(img.shape):
        subimage = img[i:i+height, j:j+width]
        if subimage.shape != monster.shape:
            continue
        try:
            mask = subimage == monster
        except ValueError:
            continue
        if all(mask[ki, kj] for ki, kj in key_inds):
            res += 1
        #
    return res

n_monsterlist = []
for img in rotflip(final):
    n_monsters = count_monsters(img, monster)
    n_monsterlist.append(n_monsters)

n_monstersquares = sum(val == "#" for val in monster.flat)
n_imagesquares = sum(val == "#" for val in final.flat)

print("*** Possible solutions depending on map orientation")
for n in sorted(set(n_monsterlist)):
    roughness = n_imagesquares - n * n_monstersquares
    print("n monsters = %d. Water roughness=%d" % (n, roughness))
