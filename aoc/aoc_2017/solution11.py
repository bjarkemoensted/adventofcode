# ..`··  ·.*  `·  . *·  • ·  ·.· ` ·+ .`* .·   + · . ·`  . * ·`  ·   ·.   .·.··`
# ·` .* ·`•·  ·*.   ·  .· + ·  `      Hex Ed ·.·       .   ·  ·.   *·   . · ·`*.
# .·· . `   · ·.+  ·   https://adventofcode.com/2017/day/11 .  ` .  ` ·  .* · .·
# · +*`· .·     ··•  * · .`       · . ·+   ·      ·`*    · .·  ·. ·  *•. ·`  ··.


def parse(s: str):
    res = s.split(",")
    return res


class Trace:
    dirs = ("n", "ne", "se", "s", "sw", "nw")
    dir2ind = {dir_: i for i, dir_ in enumerate(dirs)}

    def __init__(self, verbose=False):
        self.steps = {k: 0 for k in self.dirs}
        self.verbose = verbose
        self.max_dist = 0

    @classmethod
    def rotate(cls, direction: str, n_turns: int):
        """If facing in input direction, returns the resulting direction after rotating clockwise n times"""
        ind = cls.dir2ind[direction]
        shifted_ind = (ind + n_turns) % len(cls.dirs)
        res = cls.dirs[shifted_ind]
        return res

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    @classmethod
    def opposite(cls, step: str):
        """Takes a direction, e.g. 'sw' and returns the direction which may cancel it out, e.g. 'ne'."""
        res = cls.rotate(direction=step, n_turns=len(cls.dirs) // 2)
        return res

    @classmethod
    def iterate_compound(cls, step):
        """Takes a direction and iterates over other directions and the total direction resulting for taking a step
        in both directions. For instance, 'n' yields ('sw', 'nw') and ('se', 'ne')"""
        for shift in (+1, -1):
            far = cls.rotate(direction=step, n_turns=shift*2)
            near = cls.rotate(direction=step, n_turns=shift*1)
            yield far, near

    @property
    def dist(self):
        """Total distance to starting point given the steps recorded"""
        res = sum(self.steps.values())
        return res

    def walk_single(self, step: str):
        """Takes a single step in the specified direction"""

        self.vprint(f"Walking> {step}...")
        # Cancels out a step in the opposite direction, if any have been taken
        opposite = self.opposite(step)
        if self.steps[opposite] > 0:
            self.steps[opposite] -= 1
            return

        # Otherwise, see if we can combine two steps into a single one, e.g. 'n' + 'se' -> 'ne'
        for other, compound in self.iterate_compound(step):
            if self.steps[other] > 0:
                self.steps[other] -= 1
                self.steps[compound] += 1
                return
            #

        # Otherwise, we have to record the step
        self.steps[step] += 1

    def walk(self, steps: str | list):
        if isinstance(steps, str):
            steps = [steps]

        for step in steps:
            self.walk_single(step)
            self.max_dist = max(self.max_dist, self.dist)
        #
    #


def solve(data: str) -> tuple[int|str, int|str]:
    steps = parse(data)
    route = Trace()

    route.walk(steps)
    print(route.steps)

    star1 = route.dist
    print(f"Solution to part 1: {star1}")

    star2 = route.max_dist
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2017, 11
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
