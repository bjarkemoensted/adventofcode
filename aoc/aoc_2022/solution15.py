#  .*`·  · *·   · .• `  .··+* ·`   ·  . ··*   `·+      .·`* .  ·`* · .·`  *·`·. 
# ··`.·*    .· ·   ·•. · + ·` Beacon Exclusion Zone ` ··•    `  *.  · .• `.*· ·`
#  `·· .·+· *  ·.·*`·  https://adventofcode.com/2022/day/15   ·`· +. ·`·* ·   *·
# · .+ ·*`.·   ` +· .·   . `·   · .` · `    .·+ ·*·` .   +`·   .·  · `+.·    ·`·

import re
from dataclasses import dataclass, field
from typing import Iterator, Self


@dataclass(frozen=True, eq=True)
class Point:
    """Represents a 2D point with x/y coordinates.
    Has helper methods for computing distance, and for converting between
    lattice coordinates and regular coordinates.
    Lattice coordinates are practical because diamond shapes
    (areas with distance <= some radius from a point) in regular space become
    squares in lattice space, making it much simple to calculate overlaps, etc."""

    x: int
    y: int

    def dist(self, other) -> int:
        """Compute the distance to another point"""
        if not isinstance(other, Point):
            return NotImplemented
        res = abs(self.x - other.x) + abs(self.y - other.y)
        return res
    
    def shift(self, dx=0, dy=0) -> Self:
        res = self.__class__(x=self.x+dx, y=self.y+dy)
        return res
    
    def to_lattice_coords(self) -> Self:
        """Convert point form regular to lattice coordinates"""
        res = self.__class__(x=self.x+self.y, y=self.x-self.y)
        return res
    
    def to_regular_coords(self) -> Self:
        """Convert a point from lattice coordinates to regular coordinates"""
        if (self.x + self.y) % 2 != 0:
            raise ValueError("coords mod 2 must be zero to correspond to int. coords")
        
        x = (self.x + self.y) // 2
        y = (self.x - self.y) // 2
        res = self.__class__(x=x, y=y)
        return res


@dataclass
class Sensor:
    pos: Point
    beacon: Point
    radius: int = field(init=False)

    def __post_init__(self) -> None:
        self.radius = self.pos.dist(self.beacon)


def parse(s: str) -> list[Sensor]:
    pattern = r"Sensor at x=(-?\d+), y=(-?\d+): closest beacon is at x=(-?\d+), y=(-?\d+)"
    res = []
    for line in s.splitlines():
        m = re.match(pattern, line)
        assert m is not None
        x_sensor, y_sensor, x_beacon, y_beacon = map(int, m.groups())
        sensor = Sensor(
            pos=Point(x=x_sensor, y=y_sensor),
            beacon=Point(x=x_beacon, y=y_beacon)
        )
        res.append(sensor)

    return res


def partition_spans(*spans: tuple[int, int]) -> list[tuple[int, int]]:
    """Converts input tuples representing intervals (low, high), both inclusive,
    into a list of mutually exclusive spans containing the same values."""

    res = []
    ordered = sorted(spans)
    low, high = ordered[0]
    for i in range(1, len(ordered)):
        a, b = ordered[i]
        split = a > high + 1
        if split:
            res.append((low, high))
            low, high = a, b
        else:
            high = max(b, high)
        #    
    res.append((low, high))

    return res


def scan(
        squares: list[tuple[tuple[int, int], tuple[int, int]]]
    ) -> Iterator[tuple[tuple[int, int], list[tuple[int, int]]]]:
    """Takes a list of squares,represented like ((x_low, x_high), (y_low, y_high)),
    and scans across x coordinates, determining the intervals of y-values
    that are inside a square at the given x coordinate.
    At every x coordinate where these y-intervals change, this iterator returns
    a tuple of (x_previous, x_now), and a list of y-intervals, each represented
    as a tuple, e.g. (y_low, y_high)."""

    # x-coordinates where each square begins or ends
    events = []
    for square in squares:
        (x0, x1), (y0, y1) = square
        events.append((x0, y0, y1))
        events.append((x1, y0, y1))
    
    # Keep track of currently active y-intervals, and previous x coord
    active: set[tuple[int, int]] = set()
    x_prev: int|None = None

    for x, y_low, y_high in sorted(events):
        span = y_low, y_high
        if x_prev is not None:
            limits = partition_spans(*active)
            yield (x_prev, x), limits

        if span in active:
            active.remove(span)
        else:
            active.add(span)
        x_prev = x


def count_excluded(sensors: list[Sensor], y: int) -> int:
    """Counts the number of x-coordinates on the specified line, where
    no beacon can be located."""
    
    spans = []
    for sensor in sensors:
        # Skip if the scanner is further from the line than its radius
        dist = abs(y - sensor.pos.y)
        if dist > sensor.radius:
            continue
        
        # Determine the span of x-values covered by the scanner
        width = (sensor.radius - dist)
        span = (sensor.pos.x - width, sensor.pos.x + width)
        spans.append(span)

    # Count the points covered by at least one scanner    
    spans = partition_spans(*spans)
    n_points = sum(high + 1 - low for low, high in spans)

    # Subtract the points where a beacon is located on the line
    x_beacons = {sensor.beacon.x for sensor in sensors if sensor.beacon.y == y}
    n_points -= sum(any(low <= x <= high for low, high in spans) for x in x_beacons)
    return n_points


def compute_tunning_frequency(p: Point) -> int:
    """Computes the tuning frequency of the distress beacon"""
    res = p.x*4000000 + p.y
    return res


def determine_distress_location(sensors: list[Sensor], limit: int):
    """Determines the location of the distress beacon, given the input sensor
    data. This function determines the diamond shapes (Manhatten distance circles)
    around each sensor, inside which we can exclude the beacon being located.
    These regions are converted into lattice coordinates (x, y -> u, v), where they appear
    as squares. We can then scan across the transformed values looking for
    a 'hole' (e.g. if at a given transformed u-cordinate u=10, the active v-regions
    are [(1, 17), (19, 30)]), the distress beacon is at lattice coordinates u, v = 10, 18,
    and the location can be found by transforming back into regular coordinates."""

    # Determine the square in lattice space around each sensor
    squares = []
    for sensor in sensors:
        # Take the leftmost corner of the diamond shape in x, y coords
        corner = sensor.pos.shift(dx=-sensor.radius)
        cp = corner.to_lattice_coords()
        # Take the opposite corner as well, and store the intervals of the lattice coordinates
        opposite = cp.shift(2*sensor.radius, 2*sensor.radius)
        square = ((cp.x, opposite.x), (cp.y, opposite.y))
        squares.append(square)

    # Candidate coordinates for the distress beacon
    candidates: set[Point] = set()

    for (u_low, u_high), ranges in scan(squares):
        # If we have a single, coherent interval, there's nowhere for the beacon to hide
        if len(ranges) == 1:
            continue
        
        for i in range(1, len(ranges)):
            # Check the values between intervals
            _, below = ranges[i-1]
            above, _ = ranges[i]
            for u in range(u_low+1, u_high):
                for v in range(below, above):
                    # lattice coordinates with remainder mod 2 don't correspond to real integer points
                    if (u + v) % 2 != 0:
                        continue
                    
                    # Convert back into regular coordinates
                    cand = Point(u, v).to_regular_coords()
                    # Check bounds
                    in_bounds = all(coord <= limit for coord in (cand.x, cand.y))
                    if not in_bounds:
                        continue
                    
                    # Double check that we're not in range of a sensor
                    assert all(sensor.pos.dist(cand) > sensor.radius for sensor in sensors)
                    candidates.add(cand)
                #
            #
        #
    
    # Make sure the result isn't ambiguous
    assert len(candidates) == 1
    res = candidates.pop()
    return res


def solve(data: str) -> tuple[int|str, ...]:
    pairs = parse(data)

    star1 = count_excluded(pairs, y=2_000_000)
    print(f"Solution to part 1: {star1}")

    loc = determine_distress_location(pairs, limit=4_000_000)
    star2 = compute_tunning_frequency(loc)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 15
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
