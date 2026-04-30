# ·.`* `+·    ·· *+  ·   *  ·`   ·*  ·.   ·`·+  *· ·  +.  *·`   ·. * ·`·  *· .*·
#   ·       ·*..*`·   · ` ·+ . ·` Oxygen System  . ·   · *·   *·  · `  +.· .  ·*
# .·   ·*.` + `*·   ·  https://adventofcode.com/2019/day/15   ·.• `  *·  .·* `.·
# ·.•·`+ *  `· · +.· *` .·* `· + ·      ·*   ·`* ·. ·  *·.  .·  `· ·  •.·+  ·*`.

from collections import defaultdict, deque
from enum import IntEnum
from typing import Iterator, Self

from aoc.aoc_2019.intcode import Computer

type coord = tuple[int, int]


class Direction(IntEnum):
    NORTH = 1
    SOUTH = 2
    WEST  = 3
    EAST  = 4


_dir_order = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
_dir_rev = {_dir_order[i]: _dir_order[(i+2) % len(_dir_order)] for i in range(len(_dir_order))}


class StatusCode(IntEnum):
    HIT_WALL = 0
    MOVED = 1
    FOUND_OXYGEN = 2


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


def step_dir(pos: coord, dir_: Direction) -> coord:
    i, j = pos
    match dir_:
        case Direction.NORTH:
            return i-1, j
        case Direction.SOUTH:
            return i+1, j
        case Direction.EAST:
            return i, j+1
        case Direction.WEST:
            return i, j-1
        case _:
            raise ValueError
        #
    #


class Layout:
    ORIGIN: coord = (0, 0)

    def __init__(self, program: list[int]) -> None:
        # Variables for keeping track of the layout
        self.neighbors: dict[coord, set[coord]] = defaultdict(set)
        self.oxygen_system: coord|None = None
        self.walls: set[coord] = set()

        # Stuff for keeping track of the repair drone
        self.computer = Computer(program).run()
        assert not self.computer.stdout
        self.visited: set[coord] = set()
        self.pos = self.ORIGIN

        # Let the repair drone explore the map
        self._explore()

    def move_drone(self, dir_: Direction) -> StatusCode:
        """Attempt to move drone. Update current position if succesful"""
        assert not self.computer.stdout
        assert not self.computer.stdin
        status = self.computer.add_input(dir_).run().read_stdout()
        assert not self.computer.stdout

        if status != StatusCode.HIT_WALL:
            self.pos = step_dir(self.pos, dir_)

        return StatusCode(status)

    def _explore(self) -> Self:
        """DFS exploration using the repair drone. From every site visited, the drone explores
        every non-visited neighbor, then backtracks to its previous location"""
        
        assert self.pos not in self.visited
        self.visited.add(self.pos)
        old_pos = self.pos

        for dir_ in Direction:
            # Check if moving in the direction hits an already explored site
            neighbor = step_dir(self.pos, dir_)
            if neighbor in self.walls or neighbor in self.visited:
                continue

            # Move the drone
            status = self.move_drone(dir_)

            # If it hits a wall, register the wall's location and try another direction
            if status == StatusCode.HIT_WALL:
                self.walls.add(neighbor)
                continue
            
            # Otherwise, add a link between position after and before moving
            self.neighbors[old_pos].add(self.pos)
            self.neighbors[self.pos].add(old_pos)

            # Note the location if we found the oxygen system
            if status == StatusCode.FOUND_OXYGEN:
                assert self.oxygen_system is None
                self.oxygen_system = neighbor
            
            # Recurse from the new position
            self._explore()
            # Backtrack after done exploring from this position
            self.move_drone(_dir_rev[dir_])
            
            #
        return self

    def _iter_bfs(self, startat: coord) -> Iterator[tuple[int, coord]]:
        """Iterates from the starting point, returning each point encountered,
        and its distance to the starting point"""
        
        visited = {startat}
        queue = deque([(0, startat)])

        while queue:
            d, u = queue.popleft()
            yield d, u

            for v in self.neighbors[u]:
                if v in visited:
                    continue
                visited.add(v)
                queue.append((d+1, v))
            #
        #
    
    def distance(self, source: coord, target: coord) -> int:
        """Returns the distance from one site to another"""
        path = self._iter_bfs(startat=source)
        res, _ = next((dist, node) for dist, node in path if node == target)
        return res

    def max_dist_from_point(self, source: coord) -> int:
        """Returns the maximum distance to any point starting from the specified source point.
        Oxygen fills one site per time, so the time to fill is the
        greatest distance from the oxygen system."""
        
        dists = (d for d, _ in self._iter_bfs(startat=source))
        res = max(dists)
        return res


def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)

    G = Layout(program)
    target = G.oxygen_system
    assert target is not None
    star1 = G.distance(source=G.ORIGIN, target=target)
    print(f"Solution to part 1: {star1}")

    star2 = G.max_dist_from_point(source=target)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 15
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
