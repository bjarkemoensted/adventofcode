# .`*·`·    ·`.*·· .    + `·   · *.` ·*`.  · ·`. *` ·  `   *. ··` *·`.  *` .·`·*
# `  +·  `· *. ·  ` · .`* ·+ ·`.`  Giant Squid ·+·    +.·`*·          ` ·.`·+. ·
# *··`.+.·    * ` ·  · https://adventofcode.com/2021/day/4  + `* ··  · +.*·`.·+ 
# · `·+`·.*. ·   `    *·`.     ·  * ·` .·`  ·* `    ·` · *` ·* ·. . ·  *`·*  .`·

from typing import Iterator

import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> tuple[list[int], list[NDArray[np.int_]]]:
    parts = s.split("\n\n")
    # Read in numbers for bingo games
    numbers = [int(x) for x in parts[0].split(",")]

    # Parse bingo boards in 5x5 numpy arrays
    boards = []
    for boardstring in parts[1:]:
        board = np.array([[int(s.strip()) for s in line.strip().split()] for line in boardstring.split("\n")])
        boards.append(board)

    return numbers, boards


class Game:
    """Game class to keep track of progress of a bingo game"""

    def __init__(self, board) -> None:
        self.checked = np.zeros_like(board)  # 1=checked, 0=unchecked
        self.board = board
        # Map each board number to its position on board
        self.positions = {}
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                self.positions[val] = (i, j)
            #
        #

    def check(self, number) -> None:
        """If number exists on board, check it"""
        if number in self.positions:
            i, j = self.positions[number]
            self.checked[i, j] = 1
        #

    def is_filled(self) -> bool:
        """Check if the board is filled (one row or column filled)"""
        res = False

        nrows, ncols = self.checked.shape
        if any(val == nrows for val in np.sum(self.checked, axis=0)):
            res = True
        elif any(val == ncols for val in np.sum(self.checked, axis=1)):
            res = True
        return res

    def get_unchecked_numbers(self) -> list[int]:
        """Returns list of numbers that aren't checked"""
        unchecked_mask = np.logical_not(self.checked).astype(int)
        res = [val for val in np.multiply(unchecked_mask, self.board).flat if val != 0]
        return res
    #


def final_scores(numbers: list[int], boards: list[NDArray[np.int_]]) -> Iterator[int]:
    """Determines the final scores for each board as the numbers are used.
    Returns final scores in the order the boards complete."""
    
    games = [Game(board) for board in boards]
    games.reverse()
    
    for number in numbers:
        # Go over games and indices in reverse, so we can delete completed games while iterating
        for i, game in reversed(list(enumerate(games))):
            game.check(number)
            if game.is_filled():
                # Game is completed. Compute score and return it
                unchecked = game.get_unchecked_numbers()
                score = number * sum(unchecked)
                yield score
                del games[i]
            #
        #
    #


def solve(data: str) -> tuple[int|str, ...]:
    numbers, boards = parse(data)
    scores = list(final_scores(numbers, boards))

    star1 = scores[0]
    print(f"Solution to part 1: {star1}")

    star2 = scores[-1]
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 4
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
