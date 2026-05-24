# `.·*· .·`   `·. *  ` ·  ··`  .·+  ·`* · ·.  ·.`*  ·   +·  .· *` ·. · · .+ ··*·
# ·*`·  • ·. `   ··.*   ·  *· `  .  Cryostasis     +·· `.  *· ·.*     .`* ··`.·`
# ·`· +·`  * •·  . `·  https://adventofcode.com/2019/day/25  `. ·  · .* ·`    ··
# .·.    ` ·· * · .   ·`      · ··  .*` ·*   .`+··.     · ·`   ·  ` · ·   .` · *

import re
from collections import defaultdict, deque
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError
from copy import deepcopy
from dataclasses import dataclass, field
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

    def take(self, *items: str) -> Self:
        """Take an item"""
        for item in items:
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

    def can_take_item(self, item: str) -> bool:
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
        #
    #


class Starship:
    DIRECTION_KEY = "door"
    TARGET_ROOM = "Pressure-Sensitive Floor"

    def __init__(self, program: list[int]) -> None:
        self.program = [v for v in program]
        self.G = nx.DiGraph()

        # Keep track of where items are located
        self.items: dict[str, str] = dict()
        self.traps: set[str] = set()  # items that shouldn't be picked up

        self.explore()
        # cache all dists and paths
        self.paths = {u: d for u, d in nx.all_pairs_dijkstra_path(self.G)}
        self.dists = {u: {v: len(path)-1 for v, path in d.items()} for u, d in self.paths.items()}

    def determine_path(self, source: str, target: str) -> Iterator[str]:
        """Determines the instructions ('north', 'south', etc.)
        which form the shortest path from source to target."""

        assert all(node in self.G for node in (source, target))
        path = self.paths[source][target]
        if not path:
            return
        
        for i, u in enumerate(path[:-1]):
            v = path[i+1]
            direction = self.G[u][v][self.DIRECTION_KEY]
            yield direction

    def _optimal_route(self, source: str, target: str, *must_visit: str) -> tuple[str, ...]:
        """Determine the shortest path from source to target which visites every required node"""

        required_set = set(must_visit)
        # queue of current_path, current length
        s0 = (source, frozenset({source}))
        stack: list[tuple[str, frozenset[str]]] = [s0]

        best: dict[tuple[str, frozenset[str]], float] = defaultdict(lambda: float("inf"))
        best[s0] = 0
        camefrom: dict[tuple[str, frozenset[str]], tuple[str, frozenset[str]]] = dict()

        while stack:
            key = stack.pop()
            head, visited = key
            missing_nodes = sorted(required_set - visited) or [target]
            current_length = best[key]
            for node in missing_nodes:
                new_length = current_length + self.dists[head][node]
                new_visited = visited | {node}
                new_key = (node, new_visited)
                improved = new_length < best[new_key]
                if improved:
                    best[new_key] = new_length
                    camefrom[new_key] = key
                    stack.append(new_key)
                #
            #
        
        target_key = (target, frozenset(required_set | {source, target}))
        if target_key not in best:
            raise RuntimeError
        
        rev = [target_key]
        while rev[-1] in camefrom:
            rev.append(camefrom[rev[-1]])
        path = tuple(node for node, _ in reversed(rev))
        return path
    
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
        
        n_moves = 0
        visited: set[str] = set()
        queue = deque([self.initial_state()])
        # For background processes figuring out which items can be picked up
        pending_items: dict[str, Future] = dict()

        with ProcessPoolExecutor() as executor:
            while queue:
                state = queue.popleft()
                pos = state.location

                if pos in visited:
                    continue
            
                visited.add(pos)

                if pos not in self.G.nodes():
                    self.G.add_node(pos)
                
                for item in state.output.items:
                    assert item not in self.items
                    self.items[item] = pos
                    # Start backgroun job to attempt to pick up item
                    pending_items[item] = executor.submit(deepcopy(state).can_take_item, item=item)
                
                for door in state.output.doors:
                    already_done = any(v for v, d in self.G[pos].items() if d[self.DIRECTION_KEY] == door)
                    if already_done:
                        continue
                    new_state = deepcopy(state)
                    new_state.move(door)
                    n_moves += 1

                    other_room = new_state.output.labels[0]
                    self._add_edge(u=pos, v=other_room, direction=door)
                    queue.append(new_state)
                #
            for item, job in pending_items.items():
                if job.result() is False:
                    self.traps.add(item)
                #
            #
        #

    def _move_to(self, state: State, target: str) -> None:
        path = self.determine_path(source=state.location, target=target)
        state.move(*path)

    def _try_items(self, state: State) -> Literal["heavier", "lighter"]|int:
        """Tries stepping on the pressure plates with the specified items.
        Returns whether the system responds that other droids are lighter/heavier,
        or if we hit the right weight, determine the provided keypad code
        from the terminal output."""

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
   
        initial_state = self.initial_state()
        
        # Move to the room adjacent to the pressure plates
        next_to_target = next(self.G.neighbors(self.TARGET_ROOM))
        item_locs = sorted(v for k, v in self.items.items() if k not in self.traps)

        item_locs = sorted(v for k, v in self.items.items() if k not in self.traps)
        next_to_target = next(self.G.neighbors(self.TARGET_ROOM))
        path = self._optimal_route(initial_state.location, next_to_target, *item_locs)

        for room in path:
            self._move_to(initial_state, room)
            initial_state.take(*initial_state.output.items)

        all_items = sorted(initial_state.inventory)
        all_items_set = frozenset(all_items)
        k0 = frozenset(all_items)
        state_cache = {k0: initial_state}

        def make_state(required_items: frozenset[str]) -> State:
            """Generates a state from the closest available in the cache
            (requiring the fewest take/drop ops)"""
            nonlocal state_cache
            if required_items in state_cache:
                return state_cache[required_items]
            
            # Determine closest state
            nearest = min(state_cache.keys(), key=lambda s: len(s ^ required_items))
            s_ = deepcopy(state_cache[nearest])
            # Generate target state by dropping/taking items
            s_.drop(*(nearest - required_items))
            s_.take(*(required_items - nearest))
            state_cache[required_items] = deepcopy(s_)
            return s_

        queue = deque([k0])
        seen: set[frozenset[str]] = set()

        too_heavy: list[frozenset[str]] = []
        too_light: list[frozenset[str]] = []

        while queue:
            key = queue.pop()
            if key in seen:
                continue
            seen.add(key)
            
            # If items are a larger set than a known too-large set, skip
            if any(key.issuperset(other) for other in too_heavy):
                outcome: Literal["heavier", "lighter"]|int = "lighter"
            elif any(key.issubset(other) for other in too_light):
                outcome = "heavier"
            else:
                outcome = self._try_items(make_state(key))
                if isinstance(outcome, int):
                    return outcome
                #
            
            if outcome == "lighter":  # drones are usually lighter -> too heavy
                too_heavy.append(key)
                for dropitem in sorted(key):
                    queue.append(key - {dropitem})
            elif outcome == "heavier":  # drones are usually heavier -> too light
                too_light.append(key)
                for takeitem in sorted(all_items_set - key):
                    queue.append(key | {takeitem})
                #
            #

        raise RuntimeError("No solution found")


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
