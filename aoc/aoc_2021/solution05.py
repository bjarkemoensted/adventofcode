#  ·`.   ··+`  ·`. *  ·.` ` .• ·+·  .   ·` *.· + `  · .  · `·   *·`. · *` . ·   
#    ` ·.` ·• · ·`·.*  · ·  `. Hydrothermal Venture   · ·`  . + · .+  `  ·  `··.
# `.·*    ·`· . .*`··  https://adventofcode.com/2021/day/5    ` . ·`* ·. `  ·.`*
# ··.·`.·    . `•·*`. ·     *·. ·   ` · `*.   ··  `·+``·    .  · .+·  *·  `·•`.·

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Line:
    start: tuple[int, int]
    end: tuple[int, int]
    points: tuple[tuple[int, int], ...] = field(repr=False, init=False)

    def __post_init__(self) -> None:
        """Determines the points on the line"""
        # The number of points the line contains
        n_points = max(abs(b - a) + 1 for a, b in zip(self.start, self.end))
        
        spans: list[list[int]] = []
        for low, high in zip(self.start, self.end, strict=True):
            # If line is vertical/horizontal, use the same x/y value for all points
            if low == high:
                vals = [low for _ in range(n_points)]
            else:
                step = 1 if high > low else -1
                vals = list(range(low, high + step, step))
                assert len(vals) == n_points
            spans.append(vals)

        xr, yr = spans
        self.points = tuple(zip(xr, yr, strict=True))
    
    @property
    def is_diagonal(self) -> bool:
        return all(a != b for a, b in zip(self.start, self.end))
        

def parse(s: str) -> list[Line]:
    res = []
    for line in s.splitlines():
        parts = [part.split(",") for part in line.split(" -> ")]
        start, end = ((a, b) for a, b in (tuple(map(int, p)) for p in parts))
        segment = Line(start, end)
        res.append(segment)
    
    return res


def count_overlaps(*lines: Line) -> int:
    """Counts the number of overlapping points in the input lines"""
    counts = Counter(p for line in lines for p in line.points)
    res = sum(v > 1 for v in counts.values())
    return res


def solve(data: str) -> tuple[int|str, ...]:
    lines = parse(data)
    
    star1 = count_overlaps(*(line for line in lines if not line.is_diagonal))
    print(f"Solution to part 1: {star1}")

    star2 = count_overlaps(*lines)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 5
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
