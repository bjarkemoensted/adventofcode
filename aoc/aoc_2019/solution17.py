# ·`*+·.·    `*  .·       `··.· *.  ·`   · .    *·  ·` · `.·*  ·  ·. ·+.  • `.··
# .·`·`·    ` .*`  ·     ·     +` Set and Forget  `·*  .·.`* ·     · *`· . .·  `
# `*·  ` ·.·*·  ·  .`· https://adventofcode.com/2019/day/17   · +·.`    +·· *·. 
# ·`.  * .· ·   *·+  `· .· *  . ·   *·+   ·`•. *·.·  *·`  ·.   `·  *·.* ·+`·.`··

from enum import StrEnum

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from aoc.aoc_2019.intcode import Computer

type coord = tuple[int, int]


class Direction(StrEnum):
    UP = "^"
    DOWN = "v"
    LEFT = "<"
    RIGHT = ">"


class Symbol(StrEnum):
    SCAFFOLD = "#"
    SPACE = "."
    ROBOT_FALLING = "X"
    ROBOT_UP = Direction.UP.value
    ROBOT_DOWN = Direction.DOWN.value
    ROBOT_LEFT = Direction.LEFT.value
    ROBOT_RIGHT = Direction.RIGHT.value


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


def parse_camera_output(program: list[int]) -> NDArray[np.str_]:
    """Parse the camera output from the intcode program into a numpy
    array of ASCII characters which make up the map"""
    
    output = Computer(program=program).run().read_stdout(n=-1)
    raw = "".join(map(chr, output)).strip()
    res = np.array([list(line) for line in raw.splitlines()])
    
    return res


def _build_graph(node_coords: set[coord]) -> nx.Graph[coord]:
    """Builds a networkx graph from a set of coordinates. Adjacent coordinates are connected."""
    G = nx.Graph()
    
    for u in node_coords:
        i, j = u
        for v in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
            if v in node_coords:
                G.add_edge(u, v)
            #
        #

    return G


class Scaffold:
    def __init__(self, program: list[int]) -> None:
        self.program = program.copy()
        m = parse_camera_output(program)
        
        # Determine coordinates with scaffolding, and build a graph over the coordinates
        self.node_coords = {(int(i), int(j)) for i, j in np.argwhere(m != Symbol.SPACE)}
        self.G = _build_graph(self.node_coords)
        self.intersections = tuple(node for node, degree in self.G.degree() if degree == 4)
        self.endpoints = tuple(node for node, degree in self.G.degree() if degree == 1)
        self.junctions = set(self.intersections) | set(self.endpoints)

        # Determine the initial position and direction of the vacuum robot
        robot_candidates = np.argwhere((m != Symbol.SPACE) & (m != Symbol.SCAFFOLD))
        assert len(robot_candidates) == 1
        i0, j0 = map(int, robot_candidates[0])
        self.initial_position = (i0, j0)
        self.initial_direction: Direction = Direction(m[i0, j0])
        
        # Store an ASCII map of the scaffolding  # TODO DO WE NEED THIS? !!!
        m[i0, j0] = Symbol.SCAFFOLD.value
        self.ascii_map = m

    def alignment_parameters(self) -> list[int]:
        """Return the alignment parameters for each intersection point"""
        res = [i*j for i, j in self.intersections]
        return res

    def display(self) -> None:
        m = self.ascii_map.copy()
        for i, j in self.intersections:
            m[i, j] = "O"
        
        for i, j in self.endpoints:
            m[i, j] = "X"
        
        m[*self.initial_position] = self.initial_direction
        print("\n".join(("".join(line) for line in m)), end="\n\n")


def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)
    scaffold = Scaffold(program)

    star1 = sum(scaffold.alignment_parameters())
    print(f"Solution to part 1: {star1}")

    scaffold.display()

    star2 = -1
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 17
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
