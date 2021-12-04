import numpy as np

with open("input04.txt") as f:
    parts = f.read().split("\n\n")

# Read in numbers for bingo games
numbers = [int(x) for x in parts[0].split(",")]

# Parse bingo boards in 5x5 numpy arrays
boards = []
for boardstring in parts[1:]:
    board = np.array([[int(s.strip()) for s in line.strip().split()] for line in boardstring.split("\n")])
    boards.append(board)


class Game:
    """Game class to keep track of progress of a bingo game"""
    def __init__(self, board):
        self.checked = np.zeros(shape=board.shape)  # 1=checked, 0=unchecked
        self.board = board
        # Map each board number to its position on board
        self.positions = {}
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                self.positions[val] = (i, j)
            #
        #

    def check(self, number):
        """If number exists on board, check it"""
        if number in self.positions:
            i, j = self.positions[number]
            self.checked[i, j] = 1
        #

    def is_filled(self):
        """Check if the board is filled (one row or column filled)"""
        res = False
        nrows, ncols = self.checked.shape
        if any(val == nrows for val in np.sum(self.checked, axis=0)):
            res = True
        elif any(val == ncols for val in np.sum(self.checked, axis=1)):
            res = True
        return res

    def get_unchecked_numbers(self):
        """Returns list of numbers that aren't checked"""
        unchecked_mask = np.logical_not(self.checked).astype(int)
        res = [val for val in np.multiply(unchecked_mask, self.board).flat if val != 0]
        return res


# Play games until we have a winning board
games = [Game(board) for board in boards]
winner = None
for number in numbers:
    for game in games:
        game.check(number)
    winners = [game for game in games if game.is_filled()]
    if winners:
        assert len(winners) == 1
        winner = winners[0]
        break
    #

sum_unchecked = sum(winner.get_unchecked_numbers())
star1 = sum_unchecked * number
print(f"Solution to star 1: {star1}.")

# Play games until every board has won and find the last board to win
games2 = [Game(board) for board in boards]
last_winner = None
for number in numbers:
    for game in games2:
        game.check(number)
    # Find boards that still haven't won after this round
    games_remaining = [game for game in games2 if not game.is_filled()]
    if not games_remaining:
        assert len(games2) == 1
        last_winner = games2[0]
        break
    else:
        games2 = games_remaining
    #

star2 = number * sum(last_winner.get_unchecked_numbers())
print(f"Solution to star 2: {star2}.")
