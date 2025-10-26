#  `·+ ·.  * ·`.   ·`* +·. ` *   ·`* . ·  ·  ·* .` .*·    ·`  .·*  `·* .  •`·. ·
# `·`.  ·`  *• ·   `·  .*· • .   Restroom Redoubt  ·   * ·+ `  •.  . ·` ·    *.`
# •.+· ` *· ·. •·*` +  https://adventofcode.com/2024/day/14  `·*  ·  ` •  .·`+·.
# · ·`+*`· . *`     .·     ·+ `·   `.•   ·`.·*  ·   ·`*·   `· *.`· `*.··`  . ·`*


from collections import Counter
from functools import cache


def parse(s: str):
    """Parses into list of dicts mapping 'p' and 'v' to position and velocity."""
    res = []
    
    for line in s.splitlines():
        d = dict()
        for path in line.split():
            k, stuff = path.split("=")
            t = tuple(map(int, stuff.split(",")))
            d[k] = t

        res.append(d)

    return res


class Robot:
    def __init__(self, p: tuple, v: tuple, xy_bounds: tuple):
        self.p = list(p)
        self.v = v
        self.xy_bounds = xy_bounds
    
    
    def step(self, n_steps: int):
        """Takes n steps forward (with periodic boundary conditions)"""
        for i in range(len(self.p)):
            self.p[i] = (self.p[i] + n_steps*self.v[i]) % self.xy_bounds[i]

    @property
    def quadrant(self):
        """Returns a tuple representing the robot's qudrant. Returns .e.g (1, -1) for the lower left quadrant.
        Values between quadrants get a zero, so e.g. (0, 1) is in the right half, but in the middle between
        top and bottom parts."""

        assert all(dim % 2 != 0 for dim in self.xy_bounds)
        mids = [dim // 2 for dim in self.xy_bounds]
        res = tuple((x > cut) - (x < cut) for x, cut in zip(self.p, mids))
        return res
    #


class Swarm:
    def __init__(self, robot_data: list, width: int, height: int):
        self.xy_bounds = (width, height)
        self.robots = [Robot(**d, xy_bounds=self.xy_bounds) for d in robot_data]
    
    def step(self, n:int=1):
        """Have all robots take one or more steps forward"""
        for robot in self.robots:
            robot.step(n)
        #
    
    def as_string(self, pad="", empty=".", robot=None):
        """Helper method for displaying the robots"""
        counts = Counter(r.p for r in self.robots)
        n_digits = len(str(max(counts.values())))
        assert n_digits == 1
        lines = []
        
        cols, rows = self.xy_bounds
        
        for i in range(rows):
            line = []
            for j in range(cols):
                n = counts[(j, i)]
                char = empty
                if n != 0:
                    char = robot if robot is not None else str(n)
                line.append(char)
            lines.append(pad.join(line))
        
        res = "\n".join(lines)
        return res

    def __str__(self):
        return self.as_string()

    def iter_lines(self):
        """Iterates over lengths of coherent vertical lines"""
        
        # Keep track of running x/y coordinates, and line length
        xr = None
        yr = None
        n = 0
        
        for x, y in sorted(r.p for r in self.robots):
            # If coordinate is unchanged, ignore (due to 2 robots sharing the same position)
            if x == xr and y == yr:
                continue
        
            # We're still in the same line segment if the column (x) is unchanged, and row changed by 1
            same = x == xr and y - yr <= 1
            if not same:
                yield n
                n = 0
            
            # Update running coordinates and segment length
            xr, yr = x, y
            n += 1
        
        # If we reached the end, also yield the final segment
        if n:
            yield n
            #
        #   

    def scan_until_shape(self):
        """Repeatedly has all robots take a step until they seem to form a shape.
        Shapes are detected by looking for long vertical line segments"""
        
        n = 0
        vline_target_len = 30
        
        while max(self.iter_lines()) < vline_target_len:
            self.step()
            n += 1
            
        return n
        
    def safety_score(self):
        """Computes the safety score (product of number of robots in each quadrant)"""
        
        res = 1
        
        quadrants = Counter(r.quadrant for r in self.robots)
        for q, n in quadrants.items():
            # Exclude robots on either of the middle lines separating the quadrants
            if any(elem == 0 for elem in q):
                continue

            res *= n
        
        return res


def solve(data: str) -> tuple[int|str, int|str]:
    data = parse(data)
    
    # Set the shape (use a small one for small numbers of robot, as in the example)
    shape = (7, 11) if len(data) < 20 else (103, 101)
    height, width = shape
    
    swarm = Swarm(robot_data=data, width=width, height=height)
    swarm.step(n=100)
    star1 = swarm.safety_score()
    print(f"Solution to part 1: {star1}")
    
    swarm = Swarm(robot_data=data, width=width, height=height)
    star2 = swarm.scan_until_shape()
    print(f"Solution to part 2: {star2}")
    
    return star1, star2


def main() -> None:
    year, day = 2024, 14
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()