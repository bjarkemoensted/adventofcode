# · ·*. .  + · * `·   ·    *+·.`· .  ·`  +`  . * `   ·`·   . ·  .·*+ ·    · ·`+.
# ·`.•   `.     ·*   ·`·• `.  * . · Sand Slabs  `*.· •     ·`  · .  *•.`·   ` · 
#  . ··`* `.  `·       https://adventofcode.com/2023/day/22 . · +  *.`· `    · `
# +· . *` •·  ·   *· . `   ·  ·   *  . · ·` +     ·     *`· •     .`·  *·.`  *.·

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import typing as t

intervaltype: t.TypeAlias = tuple[int, int]


@dataclass
class Brick:
    x: intervaltype
    y: intervaltype
    z: intervaltype

    @classmethod
    def from_string(cls, s: str) -> t.Self:
        pairs = (map(int, part.split(",")) for part in s.split("~"))
        x, y, z = zip(*pairs)
        inst = cls(x=x, y=y, z=z)
        return inst

    def overlaps(self, other: Brick) -> bool:
        """Determines whether the xy region overlaps with another brick (similar to determining
        whether the planes resulting from projecting the bricks onto z=0 have an overlap)"""
        pairs = ((brick.x, brick.y) for brick in (self, other))

        for (low1, high1), (low2, high2) in zip(*pairs):
            intersect = low2 <= high1 and high2 >= low1
            if not intersect:
                return False
        
        return True

    def vertical_distance(self, other: Brick) -> int:
        """Returns the minimum vertical distance between two bricks.
        If the bricks share some z-coordinates, 0 is returned.
        Otherwise, the lowest difference between any z-coordinates is used."""

        a1, b1 = self.z
        a2, b2 = other.z
        if a1 > b2:
            return a1 - b2
        elif a2 > b1:
            return b1 - a2
        else:
            return 0
        #
    
    @property
    def height(self) -> int:
        """The height above the ground"""
        a, _ = self.z
        return a
    
    def fall(self, dist: int) -> None:
        """Moves the block down by the specified amount (modifies in-place)"""
        a, b = self.z
        self.z = (a - dist, b - dist)


def parse(s: str) -> list[Brick]:
    res = [Brick.from_string(line) for line in s.splitlines()]
    return res


class Tetris:
    """Helper class for handling bricks and doing stuff like calculating the posititions of the bricks as they fall,
    determining which bricks support one another, etc."""

    def __init__(self, bricks: t.Iterable[Brick]) -> None:
        self._bricks: dict[int, Brick] = dict()
        self._xy_overlaps: dict[int, set[int]] = defaultdict(set)

        for brick in bricks:
            self.add_brick(brick)
        #
    
    def add_brick(self, brick: Brick) -> int:
        """Add a brick. Returns the index of the brick.
        Also updates the registry of which bricks have overlapping xy-regions (to speed up
        collision testing)"""

        n = len(self._bricks)
        if n in self._bricks:
            raise RuntimeError
        
        self._bricks[n] = brick
        
        # Update registry of intersecting xy planes
        for ind, other_brick in self._bricks.items():
            if ind == n:
                continue
            
            if brick.overlaps(other_brick):
                self._xy_overlaps[n].add(ind)
                self._xy_overlaps[ind].add(n)

        return n

    def get_vertical_dists(self, ind: int) -> dict[int, int]:
        """ind: index of a brick.
        Returns a dict mapping other dict inds to how far below the input brick they are.
        Negative values are used if the other brick is above the input one."""

        brick = self._bricks[ind]
        res = dict()
        for otherind in self._xy_overlaps[ind]:
            otherbrick = self._bricks[otherind]
            dist = brick.vertical_distance(otherbrick)
            res[otherind] = dist
        
        return res

    def let_fall(self) -> None:
        """Lets the brick fall into place. Starts with bricks closer to the ground to avoid issues with
        bricks collidint mid-air"""
        ordered = sorted(self._bricks.keys(), key=lambda k: min(self._bricks[k].z))
        for ind in ordered:
            brick = self._bricks[ind]
            vdists = self.get_vertical_dists(ind)
            maxdist = min((v for v in vdists.values() if v >= 0), default=brick.height)
            assert maxdist != 0
            brick.fall(maxdist - 1)
        #

    def get_supports(self) -> dict[int, list[int]]:
        """Maps each brick index to a list of indices of bricks which rest on it.
        'rest on' here doesn't preclude resting on other bricks as well."""

        supports = {i: [] for i in self._bricks.keys()}
        for i in self._bricks.keys():
            for other, dist in self.get_vertical_dists(i).items():
                if dist == 1:
                    supports[other].append(i)
                #
            #
        return supports

    def chain_reactions(self) -> dict[int, int]:
        """Computes for each brick how many other bricks will be affected by disintegrating it."""

        # Map each brick to bricks which rest on it (so {A: [B, C]} means B and C rest upon A)
        supports = self.get_supports()

        # Conversely, map each to the bricks it rests upon (so {A: {B, C}} means A rests on B and C)
        supported_by = {i: set() for i in self._bricks.keys()}
        for k, v in supports.items():
            for otherind in v:
                supported_by[otherind].add(k)
            #
        
        res = dict()

        # Try disintegration each brick, and determine how many bricks are affected
        for i in self._bricks.keys():
            
            affected = set()
            front = {i,}  # this holds the 'new' bricks in the computation
            next_ = set()

            # Repeatedly consider any brick resting upon bricks affected so far
            while front:
                # Update with the newly affected bricks
                affected |= front
                
                # Any brick resting on a newly affected brick might in turn get affected - check those
                candidates = {touched for elem in front for touched in supports[elem]}
                for candidate in candidates:
                    # Check if every brick on which the candidate rests is now affected
                    foundation = supported_by[candidate]
                    reaction_continues = foundation.issubset(affected)
                    if reaction_continues:
                        next_.add(candidate)
                    #
                
                front = next_
                next_ = set()

            # The reaction size is the number of affected bricks, except the disintegrated one
            res[i] = len(affected) - 1

        return res


def solve(data: str) -> tuple[int|str, ...]:
    bricks = parse(data)

    tetris = Tetris(bricks)
    tetris.let_fall()
    reaction_sizes = tetris.chain_reactions()

    star1 = sum(v == 0 for v in reaction_sizes.values())
    print(f"Solution to part 1: {star1}")

    star2 = sum(reaction_sizes.values())
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 22
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
