# ·.*`··+ .`*     . ·   * .  ` · *·   .  + ·*`·`   · * +.   ·+`.   · `·*`  ·`•· 
# *` ·`    .· ·`+ `· .     `*·.  · Care Package .·  ··.  ` ·       `· .  · . ·+`
#  •` ·   ·   `· ·*`.  https://adventofcode.com/2019/day/13   .  ·   ·  * `+·`.·
# `··+.`  ·*`· . *  +· `· · .+*`  ·  `* ·      ·* .   `· .·+`·  + ·    ·.`+·* ·.

import time
from enum import IntEnum, StrEnum
from typing import Self

from aoc.aoc_2019.intcode import Computer

ESC = "\033["

class ANSI(StrEnum):
    CLEAR_SCREEN = ESC + "2J"
    CURSOR_HIDE = ESC + "?25l"
    CURSOR_SHOW = ESC + "?25"


class JoystickInputs(IntEnum):
    NEUTRAL = 0
    LEFT = -1
    RIGHT = +1


class Tiles(IntEnum):
    EMPTY = 0
    WALL = 1
    BLOCK = 2
    PADDLE = 3
    BALL = 4


SYMBOLS = {
    Tiles.EMPTY: " ",
    Tiles.WALL: "█",
    Tiles.BLOCK: "#",
    Tiles.PADDLE: "-",
    Tiles.BALL: "O",
}


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


def determine_object_coords(program: list[int]) -> dict[tuple[int, int], Tiles]:
    """Starts the game and determines the coordinate for each object
    in the game"""

    res = dict()
    computer = Computer(program).run()
    while computer.stdout:
        x, y, tile_id = computer.read_stdout(n=3)
        
        tile = Tiles(tile_id)
        res[(x, y)] = tile
    
    return res


class Display:
    """A simple ASCII display, using ANSI escape codes to display content.
    Supports get/set item, so it can do stuff like
        my_display[(x, y)] = "x"
    to write a character at a specific location.
        del my_display[(x, y)]
    overwrites the coordinate with ' '
    """

    def __init__(self, dummy=False) -> None:
        self.dummy = dummy
        self.objects: dict[tuple[int, int], str] = dict()
        self.clear()
        self.max_y = 0

    def move_cursor(self, coord: tuple[int, int]):
        """Moves the cursor to the specified coordinates"""
        x, y = coord
        # Add 1 because ANSI uses index 1
        self._print(f"\033[{y+1};{x+1}H")

    def __enter__(self) -> Self:
        self._print(ANSI.CURSOR_HIDE)
        self.redraw()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.finish()

    def redraw(self) -> None:
        self.clear()
        for coord, tile in self.objects.items():
            self[coord] = tile

    def draw(self, coord: tuple[int, int], char: str, allow_multichar=False):
        """Draw a character at the specified coordinate.
        Unless allow_multichar is True, will cause an error if character doesn't
        have length one"""

        # Update max y coordinate seen. So we can move the cursor accordingly
        _, y = coord
        if y > self.max_y:
            self.max_y = y
        
        if not allow_multichar:
            assert len(char) == 1

        self.move_cursor(coord)
        self._print(char)
    
    def _print(self, *args, **kwargs) -> None:
        """Print a character. Keeping this in a central method do facilitate
        consistency with e.g. flushing, end chars, etc"""
        if self.dummy:
            return
        
        defaults = {
            "flush": True,
            "end": ""
        }
        kws = {k: v for k, v in kwargs.items()}
        for k, v in defaults.items():
            if k not in kws:
                kws[k] = v
            #
        print(*args, **kws)

    def finish(self):
        self.move_cursor((0, self.max_y + 1))
        self._print(ANSI.CURSOR_SHOW)

    def clear(self) -> Self:
        self._print(ANSI.CLEAR_SCREEN)
        return self

    def __setitem__(self, key: tuple[int, int], value: str):
        self.draw(key, value)
        self.objects[key] = value

    def __delitem__(self, key):
        self.draw(key, " ")
        self.objects.pop(key, None)


def play(program: list[int], use_display: bool=False) -> int:
    """Play the breakout game on the IntCode computer.
    use_display can be set to True to view the game being played.
    The game is 'solved' by constantly moving the paddle towards the
    ball's x-coordinate"""

    computer = Computer(program)
    computer.memory[0] = 2  # Tells the game software we inserted a coin
    
    ball = (-1, -1)
    paddle = (-1, -1)
    points = -1
    
    with Display(dummy=not use_display) as display:
        while not computer.halted:
            if use_display:
                time.sleep(.05)
            
            # Read elements to be displayed from the computer
            outputs = computer.run().read_stdout(n=-1)
            for x, y, id_ in (outputs[i: i+3] for i in range(0, len(outputs), 3)):
                # Special values of x and y means the output indicates a point update
                point_update = x == -1 and y == 0
                if point_update:
                    points = id_
                    # Display the points below the max y-coordinate
                    display.draw((5, display.max_y), str(points), allow_multichar=True)
                    continue
                
                coord = (x, y)
                # If we get a new position for the ball/paddle, update it
                if id_ == Tiles.BALL:
                    ball = coord
                elif id_ == Tiles.PADDLE:
                    paddle = coord
                
                # Display the new tiles we get from the game software
                char = SYMBOLS[Tiles(id_)]
                display[coord] = char
        
            # Adjust paddle position to follow the ball
            diff = ball[0] - paddle[0]
            move = JoystickInputs.NEUTRAL
            if diff > 0:
                move = JoystickInputs.RIGHT
            elif diff < 0:
                move = JoystickInputs.LEFT
            
            computer.add_input(move)
        #
    
    return points


def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)
    objects = determine_object_coords(program)

    star1 = sum(obj == Tiles.BLOCK for obj in objects.values())
    print(f"Solution to part 1: {star1}")

    star2 = play(program, use_display=False)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 13
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
