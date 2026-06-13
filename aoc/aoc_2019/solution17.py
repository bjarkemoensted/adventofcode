# ·`*+·.·    `*  .·       `··.· *.  ·`   · .    *·  ·` · `.·*  ·  ·. ·+.  • `.··
# .·`·`·    ` .*`  ·     ·     +` Set and Forget  `·*  .·.`* ·     · *`· . .·  `
# `*·  ` ·.·*·  ·  .`· https://adventofcode.com/2019/day/17   · +·.`    +·· *·. 
# ·`.  * .· ·   *·+  `· .· *  . ·   *·+   ·`•. *·.·  *·`  ·.   `·  *·.* ·+`·.`··

from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property
from typing import Iterator, Literal, Self, TypeGuard, get_args

import numpy as np
from numpy.typing import NDArray

from aoc.aoc_2019.intcode import Computer

type coord = tuple[int, int]
type turn = Literal["L", "R"]
type instruction = turn|int


def is_instruction(obj) -> TypeGuard[instruction]:
    return isinstance(obj, int) or obj in get_args(turn.__value__)


movement_func_symbols = ("A", "B", "C")
N_MOVEMENT_FUNCTIONS = len(movement_func_symbols)
MAX_CHARS = 20


class Direction(StrEnum):
    UP = "^"
    RIGHT = ">"
    DOWN = "v"
    LEFT = "<"


dirvecs = {
    Direction.UP: (-1, 0),
    Direction.RIGHT: (0, 1),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
}
_dirs = list(Direction)
_right_turns = {dir_: _dirs[(i+1) % len(_dirs)] for i, dir_ in enumerate(_dirs)}
_left_turns = {dir_: _dirs[(i-1) % len(_dirs)] for i, dir_ in enumerate(_dirs)}


def add_tuples(a: tuple[int, int], b: tuple[int, int], n: int=1) -> tuple[int, int]:
    i, j = a
    di, dj = b
    res = (i + n*di, j + n*dj)
    return res


class Symbol(StrEnum):
    SCAFFOLD = "#"
    SPACE = "."
    ROBOT_FALLING = "X"
    ROBOT_UP = Direction.UP.value
    ROBOT_DOWN = Direction.DOWN.value
    ROBOT_LEFT = Direction.LEFT.value
    ROBOT_RIGHT = Direction.RIGHT.value
    INTERSECTION = "O"
    ENDPONT = "*"


def combine_instructions(*instructions: tuple[instruction, ...]) -> tuple[instruction, ...]:
    """Combines 2 or more tuples of instructions. Handles combining the number of steps forward if they start and end
    in the same direction, e.g. ("L", 3), (2, "R") -> ("L", 5, "R")"""
    if not instructions:
        return ()
    
    a = instructions[0]
    for b in instructions[1:]:
        if all(len(arg) > 0 for arg in (a, b)) and isinstance(a[-1], int) and isinstance(b[0], int):
            a = a[:-1] + (a[-1]+b[0],) + b[1:]
        else:
            a = a + b
        #
    
    return a


def _to_ascii_instruction(*instructions: instruction) -> str:
    assert all(is_instruction(ins) for ins in instructions)
    s = ",".join(map(str, instructions))
    return s


def _instruction_ascii_length(*instructions: instruction) -> int:
    assert all(isinstance(ins, (int, str)) for ins in instructions)
    return len(_to_ascii_instruction(*instructions))


def follow_instructions(pos: coord, heading: Direction, *instructions: instruction) -> tuple[coord, Direction]:
    """From the input position and heading, follow the input instruction(s). Returns the final position and heading.
    This assumes that each instruction is possible without e.g. falling off the map."""

    for ins in instructions:
        if ins == "L":
            heading = _left_turns[heading]
        elif ins == "R":
            heading = _right_turns[heading]
        elif isinstance(ins, int):
            pos = add_tuples(pos, dirvecs[heading], n=ins)
        else:
            raise ValueError
        #
    return pos, heading


def _iterate_instructions(
        pos: coord,
        heading: Direction,
        *instructions: instruction
    ) -> Iterator[tuple[coord, Direction]]:
    """Iterate through instructions one step/turn at a time."""
    
    yield pos, heading

    for ins in instructions:
        # Breakdown step forward instructions into individual steps
        breakdown = (1 for _ in range(ins)) if isinstance(ins, int) else (ins,)
        for bd in breakdown:
            pos, heading = follow_instructions(pos, heading, bd)
            yield pos, heading
        #
    #


@dataclass(frozen=True)
class State:
    pos: coord
    heading: Direction
    movement_funcs: tuple[tuple[instruction, ...], ...] = ()
    movement_routine: tuple[int, ...] = ()
    visited: frozenset[coord] = frozenset()

    def __post_init__(self) -> None:
        if len(self.movement_funcs) > N_MOVEMENT_FUNCTIONS:
            raise InvalidState(f"Too many movement functions defined ({len(self.movement_funcs)})")

        main_routine = self.main_routine_ascii()
        if len(main_routine) > MAX_CHARS:
            raise InvalidState(f"Main routing ({main_routine}) too long ({len(main_routine)})")
        
        for i, ascii_ins in enumerate(self.movement_funcs):
            if len(ascii_ins) > MAX_CHARS:
                raise InvalidState(f"Movement function {i} ( {ascii_ins} ) too long")
            #

    def movement_funcs_ascii(self) -> Iterator[str]:
        for func in self.movement_funcs:
            yield _to_ascii_instruction(*func)
    
    def main_routine_ascii(self) -> str:
        res = ",".join(movement_func_symbols[i] for i in self.movement_routine)
        return res
    #


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


def _site_neighbors(p: coord) -> tuple[coord, ...]:
    i, j = p
    res = ((i+1, j), (i-1, j), (i, j+1), (i, j-1))
    return res


class InvalidState(Exception):
    """Custom exception for invalid states"""
    pass


@dataclass(frozen=True)
class Segment:
    """Represents a path segment between two points on the scaffolding.
    Stores the initial position and heading, and the instructions that trace out the path segment.
    Exposes properties for the final position and heading, and the nodes visited"""

    pos_initial: coord
    heading_initial: Direction
    instructions: tuple[instruction, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.instructions, tuple)
        assert all(isinstance(ins, (int, str)) for ins in self.instructions)

    @cached_property
    def path(self) -> tuple[tuple[coord, Direction], ...]:
        """The full path (pos/heading at every step)"""
        return tuple(_iterate_instructions(self.pos_initial, self.heading_initial, *self.instructions))

    @cached_property
    def pos_final(self) -> coord:
        res, _ = self.path[-1]
        return res

    @cached_property
    def heading_final(self) -> Direction:
        _, res = self.path[-1]
        return res
    
    @cached_property
    def nodes(self) -> frozenset[coord]:
        return frozenset(c for c, _ in self.path)

    def add_instructions(self, *instructions: tuple[instruction, ...]) -> Self:
        """Add more instructions and return the resulting path segment"""
        combined = combine_instructions(self.instructions, *instructions)
        res = self.__class__(
            pos_initial = self.pos_initial,
            heading_initial = self.heading_initial,
            instructions=combined
        )
        return res

    def __add__(self, other) -> Self:
        """Combines two path segments"""
        if not isinstance(other, Segment):
            return NotImplemented
        
        # Check that the other segment can take over where this one ends
        coord_match = other.pos_initial == self.pos_final
        dir_match = other.heading_initial == self.heading_final
        if not (coord_match and dir_match):
            raise ValueError(f"Path segments do not connect - {self} doesn't connect to {other}")
        
        # If yes, combine their instructions and return the resulting segment
        res = self.add_instructions(other.instructions)
        return res


class Scaffold:
    def __init__(self, ascii_map: NDArray[np.str_]) -> None:
        m = ascii_map.copy()
        
        # Determine the initial position and direction of the vacuum robot
        robot_candidates = np.argwhere((m != Symbol.SPACE) & (m != Symbol.SCAFFOLD))
        assert len(robot_candidates) == 1
        i0, j0 = map(int, robot_candidates[0])
        self.initial_position = (i0, j0)
        self.initial_direction: Direction = Direction(m[i0, j0])
        
        # Store an ASCII map of the scaffolding
        m[i0, j0] = Symbol.SCAFFOLD.value
        self.ascii_map = m

        # Determine coordinates with scaffolding, and build a graph over the coordinates
        self.scaffolding = {(int(i), int(j)) for i, j in np.argwhere(m != Symbol.SPACE)}
        self.G = {u: {v for v in _site_neighbors(u) if v in self.scaffolding} for u in self.scaffolding}
        self.intersections = {u for u, links in self.G.items() if len(links) > 2}
        self.endpoints = {u for u, links in self.G.items() if len(links) == 1}
        # Get all the corners (non-intersections where the path turns)
        self.corners = {
            u for u in self.scaffolding
            if len(self.G[u]) == 2 and all(a != b for a, b in zip(*self.G[u]))
        }

        # Store all 'interesting' nodes - all nodes that aren't a straight line segment
        self.nodes = self.intersections | self.corners | self.endpoints

        # Cache to determine for each node and heading which path segments lead to other nodes
        self._segments_cache: dict[tuple[coord, Direction], Segment|None] = dict()
        # Cache for all possible path segment from each pos + heading
        self._movement_func_cache: dict[tuple[coord, Direction], tuple[Segment, ...]] = dict()
        # Cache for applying movement funcs
        self._apply_movement_cache: dict[tuple[Segment, tuple[instruction, ...]], Segment|None] = dict()

    
    def display(self, *highlight_coords: coord, highlight_sym: str="*") -> None:
        m = self.ascii_map.copy()
        
        for i, j in self.intersections:
            m[i, j] = Symbol.INTERSECTION
        
        for i, j in self.endpoints:
            m[i, j] = Symbol.ENDPONT
        
        m[*self.initial_position] = self.initial_direction
        for c in highlight_coords:
            m[*c] = highlight_sym

        print("\n".join(("".join(line) for line in m)), end="\n\n")
    
    @staticmethod
    def _key(pos: coord, heading: Direction) -> tuple[coord, Direction]:
        """Makes a key for cache looking given a position and heading"""
        return (pos, heading)

    def _trace_segment(self, pos: coord, heading: Direction) -> Segment|None:
        assert pos in self.nodes

        # Start by taking a step forward
        step_forward = (1,)
        segment = Segment(pos_initial=pos, heading_initial=heading).add_instructions(step_forward)
        # Check if we fall off -> no valid segment
        falls_off = segment.pos_final not in self.scaffolding
        if falls_off:
            return None
        
        # Keep determining the next valid step until we reach another node
        while segment.pos_final not in self.nodes:
            options: tuple[tuple[instruction, ...], ...] = ((1,), ("L", 1), ("R", 1))
            next_candidates = (segment.add_instructions(steps) for steps in options)
            valid = [s for s in next_candidates if s.pos_final in self.scaffolding]
            assert len(valid) == 1
            segment = valid[0]
        
        return segment

    def get_path_segment(self, pos: coord, heading: Direction) -> Segment|None:
        """Get the path to the next node from the given position and heading.
        Returns None if none exists (i.e. if a step in the initial direction falls off)"""

        key = self._key(pos, heading)
        if key in self._segments_cache:
            return self._segments_cache[key]
        
        res = self._trace_segment(pos=pos, heading=heading)
        self._segments_cache[key] = res
        return res
    
    def _iterate_movement_funcs(self, pos: coord, heading: Direction) -> Iterator[Segment]:
        """From a starting pos + heading, generates all valid path segments from there, meaning segments
        to every reachable node"""
        
        # Instructions for continuing or turning
        connecting_instructions: tuple[tuple[instruction, ...], ...] = ((), ("L",), ("R",))
        remaining = deque([Segment(pos_initial=pos, heading_initial=heading)])
        while remaining:
            segment = remaining.pop()
            for ins in connecting_instructions:
                temp = segment.add_instructions(ins)
                outgoing = self.get_path_segment(temp.pos_final, temp.heading_final)
                if outgoing is None:
                    continue
                
                next_ = temp + outgoing
                too_long = _instruction_ascii_length(*next_.instructions) > MAX_CHARS
                if too_long:
                    continue
                
                yield next_
                remaining.append(next_)
            #
        #

    def get_valid_movement_funcs(self, pos: coord, heading: Direction) -> tuple[Segment, ...]:
        """Given a start pos + heading, determine all valid movement functions starting from there"""
        key = self._key(pos, heading)
        if key in self._movement_func_cache:
            return self._movement_func_cache[key]
        
        res = tuple(self._iterate_movement_funcs(pos, heading))
        self._movement_func_cache[key] = res
        return res

    def _attempt_add_new_movement_funcs(self, state: State) -> Iterator[State]:
        n_existing = len(state.movement_funcs)
        if n_existing == N_MOVEMENT_FUNCTIONS:
            return
        
        for add_segment in self.get_valid_movement_funcs(state.pos, state.heading):
            try:
                new_state = replace(
                state,
                pos=add_segment.pos_final,
                heading=add_segment.heading_final,
                visited = state.visited | add_segment.nodes,
                movement_funcs = state.movement_funcs + (add_segment.instructions,),
                movement_routine = state.movement_routine + (n_existing,)
            )
                yield new_state
            except InvalidState:
                # Happens if the main routine becomes too large
                pass
    
    def _apply_movement_func(
            self,
            segment: Segment,
            instructions: tuple[instruction, ...]
        ) -> Segment|None:
        """Try to use a movement function. Returns None if doing so falls off"""
        
        key = (segment, instructions)
        if key not in self._apply_movement_cache:
            path = segment.add_instructions(instructions)
            self._apply_movement_cache[key] = path if path.nodes.issubset(self.scaffolding) else None
        
        return self._apply_movement_cache[key]
    
    def _attempt_reuse_movement_funcs(self, state: State) -> Iterator[State]:
        segment = Segment(pos_initial=state.pos, heading_initial=state.heading)
        for i, func in enumerate(state.movement_funcs):
            path = self._apply_movement_func(segment, func)
            if path is None:
                continue
            
            ends_mid_segment = path.pos_final not in self.nodes
            if ends_mid_segment:
                continue
            
            try:
                next_state = replace(
                    state,
                    pos=path.pos_final,
                    heading=path.heading_final,
                    visited = state.visited | path.nodes,
                    movement_routine = state.movement_routine + (i,)
                )

                yield next_state
            except InvalidState:
                continue
        #

    def neighbor_states(self, state: State) -> Iterator[State]:
        assert all(c in self.scaffolding for c in state.visited)
        assert state.pos in self.nodes
        yield from self._attempt_reuse_movement_funcs(state)
        yield from self._attempt_add_new_movement_funcs(state)

    def determine_complete_path(self) -> State:
        """Devise a method for visiting every node on the scaffolding at least once"""
        s0 = State(pos=self.initial_position, heading=self.initial_direction)
        queue = deque([s0])
        all_coords = frozenset(self.scaffolding)

        while queue:
            state = queue.pop()
            done = state.visited == all_coords
            if done:
                return state

            for neighbor in self.neighbor_states(state):
                queue.append(neighbor)
            #
        raise RuntimeError("No solution found")


def determine_dust_amount(computer: Computer, state: State) -> int:
    """Given an IntCode computer and a state representing a path on the scaffolding,
    input the main movement routine and movement functions to the program, run it,
    and read out the result (amount of dust collected by the robot)"""
    
    # Define the main movement routine
    main_ = state.main_routine_ascii()
    computer.input_ascii(main_).run().output_ascii()

    # Define each movement function
    for func in state.movement_funcs_ascii():
        computer.input_ascii(func).run().output_ascii()
    
    # Run and retrieve output
    output_ = computer.input_ascii("n").run().read_stdout(-1)
    # It seems to re-print the entire map, so just grab the last output (the result)
    res = output_[-1]

    return res


def determine_alignment_parameters(program: list[int]) -> list[int]:
    """Return the alignment parameters for each intersection point"""
    computer = Computer(program)
    computer.memory[0] = 2

    map_str = computer.run().output_ascii()
    lines = list(map_str.splitlines())
    camera_end = lines.index("")

    map_ = np.array([list(line) for line in lines[:camera_end]])
    scaffold = Scaffold(map_)
    res = [i*j for i, j in scaffold.intersections]
    return res


def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)
    computer = Computer(program)
    computer.memory[0] = 2  # override movement logic

    # Parse camera output into an ASCII map
    map_str = computer.run().output_ascii()
    lines = list(map_str.splitlines())
    camera_end = lines.index("")
    map_ = np.array([list(line) for line in lines[:camera_end]])
    scaffold = Scaffold(map_)

    alignment_parameters = [i*j for i, j in scaffold.intersections]
    star1 = sum(alignment_parameters)
    print(f"Solution to part 1: {star1}")

    final_path = scaffold.determine_complete_path()
    star2 = determine_dust_amount(computer, final_path)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 17
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
