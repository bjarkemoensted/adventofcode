# `.·*· .·`   `·. *  ` ·  ··`  .·+  ·`* · ·.  ·.`*  ·   +·  .· *` ·. · · .+ ··*·
# ·*`·  • ·. `   ··.*   ·  *· `  .  Cryostasis     +·· `.  *· ·.*     .`* ··`.·`
# ·`· +·`  * •·  . `·  https://adventofcode.com/2019/day/25  `. ·  · .* ·`    ··
# .·.    ` ·· * · .   ·`      · ··  .*` ·*   .`+··.     · ·`   ·  ` · ·   .` · *

import random
import re
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain, combinations
from typing import Iterator, Literal, Self

import networkx as nx

from aoc.aoc_2019.intcode import Computer

DIRECTIONS = ("north", "east", "south", "west")
_dir_inv = {dir_: DIRECTIONS[(i + 2) % len(DIRECTIONS)] for i, dir_ in enumerate(DIRECTIONS)}


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


def _parse_list(s: str, leading_string: str) -> tuple[str, ...]:
    """Parses a list from IntCode output. leading string is the
    header to look for, e.g.
    Items here:
      - some item
    """

    try:
        part = s.split(leading_string)[1].split("\n\n")[0]
    except IndexError:
        return ()
    
    res = tuple(re.findall("- (.*)", part))
    return res


def _parse_room(s: str) -> tuple[str, ...]:
    """Determines the room from output.
    first: boolean indicating whther the first, rather than
    second room should be used if multiple rooms are listed.
    This happens when ousted from the pressure plate room."""

    loc_hits = re.findall(r"== (.*?) ==", s)
    if len(loc_hits) == 0:
        raise RuntimeError(f"Can't parse: {s}")

    return tuple(loc_hits)


@dataclass
class Output:
    """Parsed output from the game. This is just to structure parsed output
    in an easily accessible format."""

    name: str  # name of current room
    labels: tuple[str, ...]  # all room names mentioned
    doors: tuple[str, ...]
    items: tuple[str, ...]
    raw: str = field(repr=False)  # last raw terminal output

    @classmethod
    def parse(cls, s: str) -> Self:
        # Parse multiple room labels for if ejected from the pressure plate room
        labels = _parse_room(s=s)
        name = labels[-1]
        doors = _parse_list(s, leading_string="Doors here lead:")
        items = _parse_list(s, leading_string="Items here:")
        return cls(name=name, labels=labels, doors=doors, items=items, raw=s)
    #


class InvalidAction(Exception):
    pass


class State:
    """Current game state. Automatically updates current location and figures out
    available items/directions etc"""

    def __init__(self, program: list[int]) -> None:
        self.computer = Computer(program)
        initial_output = self.computer.run().output_ascii()
        self.output = Output.parse(initial_output)
    
    @property
    def location(self) -> str:
        return self.output.name
    
    @property
    def inventory(self) -> tuple[str, ...]:
        out = self._run("inv")
        res = tuple(_parse_list(s=out, leading_string="Items in your inventory:"))
        return res
    
    def __repr__(self):
        return f"{self.__class__.__name__}(location={self.location})"

    def _run(self, command: str) -> str:
        """Run a command. Return the output"""
        res = self.computer.input_ascii(command).run().output_ascii()

        # Update room if output parses as room
        try:
            output = Output.parse(res)
            self.output = output
        except RuntimeError:
            pass

        if "Unrecognized command." in res:
            raise InvalidAction(f"Invalid command: '{command}'")
        return res

    def take(self, item: str) -> Self:
        """Take an item"""
        output = self._run(f"take {item}")
        if "You don't see that item here." in output:
            raise InvalidAction(f"Can't take: {item}")
        return self
    
    def move(self, *directions: str) -> Self:
        """Move in the input direction."""
        
        for direction in directions:
            output = self._run(direction)
            bad = ("You can't move!", "You can't go that way.")
            if any(s in output for s in bad):
                raise InvalidAction(f"Can't move {direction}: {output}")

        return self
    
    def drop(self, *items: str) -> Self:
        """Attempt to drop an item"""
        for item in items:
            output = self._run(f"drop {item}")
            if f"You drop the {item}" not in output:
                raise InvalidAction(output)
        
        return self

    @staticmethod
    def _take_and_drop(state: State, item: str) -> None:
        """Take item, then drop it again. For checking if it causes some error.
        Static method so we can run it in a separate process, killing it if it times
        out (e.g. picking up the 'infinite loop')"""
        state.take(item).drop(item)

    def check_pickup(self, item: str) -> bool:
        """Determines whether it's possible to pick up the specified item from
        the current state. Works by taking a copy of the state, then attempting to pick
        up and drop the item. If that attempt succeeds, the item can be picked up."""

        state = deepcopy(self)
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._take_and_drop, state=state, item=item)
            try:
                # If successful, we can grab the item
                future.result(timeout=.5)
                return True
            except TimeoutError:
                # If timeout, assume we can't pick it up
                executor.terminate_workers()
                return False
            except InvalidAction:
                # Can't pick it up if doing so gives an error
                return False
            #
        #

    def start_interactive_mode(self) -> None:
        """Runs the explorer drone program in interactive mode, prompting for text input
        at each step."""

        self.computer.set_display(pipe="text_display")
        print(self.output.raw)
        while not self.computer.halted:
            input_ = input()
            output = self._run(input_)
            print(output)


class Starship:
    DIRECTION_KEY = "door"
    TARGET_ROOM = "Pressure-Sensitive Floor"

    def __init__(self, program: list[int]) -> None:
        self.program = [v for v in program]
        self.G = nx.DiGraph()

        # Keep track of where items are located
        self.items: dict[str, str] = dict()

        self.explore()

    def determine_path(self, source: str, target: str) -> Iterator[str]:
        """Determines the instructions ('north', 'south', etc.)
        which form the shortest path from source to target."""

        assert all(node in self.G for node in (source, target))
        path = nx.shortest_path(self.G, source=source, target=target)
        if not path:
            return
        
        for i, u in enumerate(path[:-1]):
            v = path[i+1]
            direction = self.G[u][v][self.DIRECTION_KEY]
            yield direction

    def initial_state(self) -> State:
        return State(self.program)

    def _add_edge(self, u: str, v: str, direction: str) -> None:
        """Adds an edge from u to v, going in the specified direction from u -> v.
        For instance, if adding an edge from 'room u' to 'room v' going 'north',
        an opposite edge from v to u going south is added as well"""
        self.G.add_edge(u, v, **{self.DIRECTION_KEY: direction})
        self.G.add_edge(v, u, **{self.DIRECTION_KEY: _dir_inv[direction]})

    def explore(self) -> None:
        """Explores the ship, building a graph of connected rooms and noting
        where each item is located"""
        visited: set[str] = set()

        def process(state: State) -> State:
            nonlocal visited
            
            pos = state.location
            if pos in visited:
                return state
            
            visited.add(pos)
            
            if pos not in self.G.nodes():
                self.G.add_node(pos)
            
            for item in state.output.items:
                assert item not in self.items
                self.items[item] = pos
            
            for door in state.output.doors:
                new_state = deepcopy(state).move(door)
                state_after = process(new_state)
                other_room = state_after.output.labels[0]
                self._add_edge(u=pos, v=other_room, direction=door)

            return state
        
        state_ = self.initial_state()
        process(state_)

    def _move_to(self, state: State, target: str) -> None:
        path = self.determine_path(source=state.location, target=target)
        state.move(*path)

    def _pickup_all(self, state: State) -> None:
        """Grab all items"""
        for item, room in self.items.items():
            self._move_to(state, room)
            
            if state.check_pickup(item):
                state.take(item)
            #
        #
    
    def _try_items(self, state: State, *items) -> Literal["heavier", "lighter"]|int:
        """Tries stepping on the pressure plates with the specified items.
        Returns whether the system responds that other droids are lighter/heavier,
        or if we hit the right weight, determine the provided keypad code
        from the terminal output."""

        drop = (item for item in state.inventory if item not in items)
        state.drop(*drop)
        self._move_to(state, target=self.TARGET_ROOM)

        hits = re.findall(
            r"Droids on this ship are (\w*?) than the detected value!",
            state.output.raw
        )

        if not hits:
            # If not too light or heavy, the combination worked. Return the keypad code
            code_matches = re.findall(r"typing (\d+) on the keypad", state.output.raw)
            assert len(code_matches) == 1
            return int(code_matches[0])
        
        assert len(hits) == 1
        s = hits[0]
        assert s == "lighter" or s == "heavier"
        return s
    
    def complete(self) -> int:
        """Picks up all items, figures out the correct combination to pass
        the pressure plates, and returns the provided keypad code"""
        state = self.initial_state()
        self._pickup_all(state)

        # Move to the room adjacent to the pressure plates
        next_to_target = next(self.G.neighbors(self.TARGET_ROOM))
        self._move_to(state, next_to_target)

        # Keep track of combinations that are too light or too heavy
        too_heavy: list[set[str]] = []
        too_light: list[set[str]] = []

        # Generate all item combinations
        items = set(state.inventory)
        combs = list(chain.from_iterable(combinations(items, r) for r in range(len(items)+1)))
        combs.sort(key = lambda _: random.uniform(0, 1))

        for keep in map(set, combs):
            # Skip combination if heavier/lighter than a known too heavy/light combination
            if any(keep.issuperset(other) for other in too_heavy):
                continue
            if any(keep.issubset(other) for other in too_light):
                continue
            
            # Try the combination and return the code if successful
            result = self._try_items(deepcopy(state), *keep)
            if isinstance(result, int):
                return result
            
            # Otherwise, add to the too heavy/light combinations
            match result:
                case "lighter":
                    too_heavy.append(keep)
                case "heavier":
                    too_light.append(keep)
                case _:
                    raise RuntimeError(f"Invalid: {result}")
                #
            #
        raise RuntimeError("No working combination of items found")    
    #


def solve(data: str) -> tuple[int|str|None, ...]:
    program = parse(data)
    starship = Starship(program)

    star1 = starship.complete()
    print(f"Solution to part 1: {star1}")

    star2 = None

    return star1, star2


def main() -> None:
    year, day = 2019, 25
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
