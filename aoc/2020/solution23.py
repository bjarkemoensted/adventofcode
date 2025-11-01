#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:20:36 2020

@author: ahura
"""

test_input = "389125467"
real_input = "137826495"

class LinkedListIsh(dict):
    def set_start(self, val):
        if val not in self:
            raise KeyError('nope')
        self.start = val

    def traverse(self, startat=None):
        if not hasattr(self, 'start'):
            raise ValueError('Set a start value before traversing')
        if startat is None:
            startat = self.start
        k = startat
        while k in self and self[k] != startat:
            yield k, self[k]
            k = self[k]
        #

    def get_next(self, val=None):
        if val is None:
            val = self.start
        return self[val]

    def pop(self, n, startat=None):
        if startat is None:
            startat = self.start

        node = startat
        dest = self[node]
        first_node_in_snippet = dest

        snippet = LinkedListIsh()

        for _ in range(n):
            node = self[node]
            dest = self[node]
            snippet[node] = dest

        for k in snippet.keys():
            del self[k]
        self[startat] = dest

        snippet.set_start(first_node_in_snippet)
        return snippet

    def plug(self, other, pos):
        node = pos
        final = self[node]
        for k, _ in other.traverse():
            self[node] = k
            node = k
        self[node] = final



class CupGame:
    def __init__(self, cups, verbose=False):
        self.cups = cups
        self._mincup = min(cups)
        self._maxcup = max(cups)
        self.round_no = 0
        self.verbose = verbose
        # Make the cycle thing to store the cups
        self.cycle = LinkedListIsh()
        for i, cup in enumerate(cups):
            nextcup = cups[(i+1)%len(cups)]
            self.cycle[cup] = nextcup

        # Set the active cup
        self.cycle.set_start(cups[0])

    def pickup(self, n=3):
        snippet = self.cycle.pop(n=n)
        return snippet

    def select_destination_cup(self):

        num = self.cycle.start - 1
        while num not in self.cycle:
            num -= 1
            if num < self._mincup:
                num = self._maxcup
            #
        return num

    def place_cups(self, snippet, destination):
        self.cycle.plug(snippet, destination)

    def select_new_cup(self):
        next_cup = self.cycle.get_next()
        self.cycle.set_start(next_cup)

    def play_round(self):
        self.round_no += 1
        if self.verbose:
            print("-- move %d --" % self.round_no)
        snippet = self.pickup()
        destination = self.select_destination_cup()
        self.place_cups(snippet, destination)
        self.select_new_cup()

    def play_rounds(self, n, print_progress=True):
        for i in range(n):
            self.play_round()
            completed = i+1
            if print_progress and completed % 1000 == 0:
                percent = 100*completed/n
                msg = "Progress: %.2f%%." % percent
                print(msg, end="\r")
            #
        if print_progress:
            print()

    def get_result_string(self):
        startat = self.cycle.get_next(1)
        nums = [k for k, _ in self.cycle.traverse(startat=startat)]
        res = "".join(map(str, nums))
        return res

### Star 1
cups = [int(elem) for elem in real_input]
game = CupGame(cups, verbose=False)
game.play_rounds(100)

print(game.get_result_string())

### Star 2
cups2 = [val for val in cups]
running = max(cups)
while len(cups2) < 1000000:
    running += 1
    cups2.append(running)

game2 = CupGame(cups2)
game2.play_rounds(10000000)

target_cups = [tup[0] for tup in list(game2.cycle.traverse(startat=1))[1:3]]
print(target_cups[0] * target_cups[1])