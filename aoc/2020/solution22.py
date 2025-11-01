#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:36:09 2020

@author: ahura
"""

from copy import deepcopy


def parse(s):
    chunks = s.split("\n\n")
    decks = []
    for chunk in chunks:
        decks.append([int(line.strip()) for line in chunk.strip().split("\n")[1:]])
    return decks

with open("input22.txt") as f:
    decks_original = parse(f.read())

### Star 1
decks = deepcopy(decks_original)
while all(len(deck) > 0 for deck in decks):
    cards = [deck.pop(0) for deck in decks]
    winner = max(enumerate(cards), key=lambda t: t[1])[0]
    cards.sort(reverse=True)
    decks[winner] += cards


winner_deck = max(decks, key=len)
score = sum((i+1)*card for i, card in enumerate(winner_deck[::-1]))
print(score)


### Star 2
class GameCounter:
    def __init__(self, start=0):
        self.val = start
    def tick(self):
        self.val += 1
        return self.val


class Game:
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __init__(self, decks, game_counter=None, verbose=True):
        if game_counter is None:
            game_counter = GameCounter()
        self.game_counter = game_counter
        self.game_no = self.game_counter.tick()
        self.verbose = verbose

        self.print("\n=== Game %d ===\n" % self.game_no)
        self.decks = [deepcopy(deck) for deck in decks]
        self.history = set([])
        self.winner = None

    def set_winner(self, winner):
        self.print("The winner of game %d is player %d.\n" % (self.game_no, (winner + 1)))
        self.winner = winner

    def play_subgame(self, subdecks):
        self.print("Playing a sub-game to determine the winner")
        subgame = Game(subdecks, game_counter=self.game_counter,
                       verbose=self.verbose)
        subgame.play()

        self.print("...anyway, back to game %d." % self.game_no)

        return subgame.winner

    def draw(self):
        cards = tuple(deck.pop(0) for deck in self.decks)
        return cards

    def determine_round_winner(self, cards):
        if all(len(deck) >= card for deck, card in zip(self.decks, cards)):
            subdecks = [deck[:card] for deck, card in zip(self.decks, cards)]
            winner = self.play_subgame(subdecks)
        else:
            winner = max(enumerate(cards), key=lambda t: t[1])[0]
        return winner

    def check_store_state(self):
        state = tuple(tuple(deck) for deck in self.decks)
        if state in self.history:
            return True
        else:
            self.history.add(state)

    def play_round(self):
        # Print status
        round_no = len(self.history) + 1
        self.print("Round %d (Game %d)" % (round_no, self.game_no))
        for i, deck in enumerate(self.decks):
            self.print("Player %d's deck: %s" % ((i+1), ", ".join(map(str, deck))))

        # Terminate game if we arrive at a previous game state
        if self.check_store_state():
            self.set_winner(0)
            return

        cards = self.draw()
        for i, card in enumerate(cards):
            self.print("Player %d plays: %d" % ((i+1), card))

        round_winner = self.determine_round_winner(cards)
        self.print("Player %d wins round %d of game %d.\n" % ((round_winner + 1), round_no, self.game_no))
        round_loser = (round_winner + 1) % 2

        # Put cards back into deck
        cards = [cards[round_winner], cards[round_loser]]
        self.decks[round_winner] += cards

        if len(self.decks[round_loser]) == 0:
            self.set_winner(round_winner)

    def play(self):
        while self.winner is None:
            self.play_round()
        #
    #

game = Game(decks=decks_original, verbose=False)
game.play()

windeck = game.decks[game.winner]
score = sum((i+1)*card for i, card in enumerate(windeck[::-1]))

print(score)