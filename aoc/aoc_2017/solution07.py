# `·  .·`*· ·.  ·*.`       * ·  `+··    ·*  · `•  ·  *.`·     ·  ·   *`.  +·. `·
# ·`*+·`· .·   + ·* ` * ·. ·   + Recursive Circus  · .· * .  ·+  *.·•· `· .*  ·`
# ·. ·` + *.  ·  .`•   https://adventofcode.com/2017/day/7   + ·  ` · . `  ·• ·.
# *· · ·.   +·`    · ·`*. ·    .·` +·.  *   · ··* .`   · *   .` ·    + ·*`  · *·


from collections import Counter
from copy import deepcopy
from functools import cache
import re


def parse(s: str):
    res = dict()
    for line in s.splitlines():
        m = re.match(r"(?P<name>\S+) \((?P<weight>\d+)\)", line)
        try:
            children = tuple(line.strip().split(" -> ")[1].split(", "))
        except IndexError:
            children = tuple(())

        name = m.group('name')
        d = dict(
            name=name,
            weight=int(m.group('weight')),
            children=children
        )
        res[name] = d
    return res


def find_bottom(programs: dict) -> str:
    """Finds the root node in nested dicts (assumes one exists)"""

    children = set(sum([list(v['children']) for v in programs.values()], []))
    parentless = set(programs.keys()) - children
    assert len(parentless) == 1
    res = list(parentless)[0]
    return res


def iterate_nodes(programs: dict, start: str):
    """Iterates over nodes in a nested dict"""
    nodes = [start]
    while nodes:
        next_ = []
        for node in nodes:
            yield node
            next_ += list(programs[node]["children"])

        nodes = next_


def totals(programs: dict):
    """Given the program data, provides a cached method for computing the total weights, i.e. the recursive sum
    of node weights."""

    programs = deepcopy(programs)

    @cache
    def determine_total_weight(node):
        """Making this wrapped so we can use caching (dicts aren't hashable)"""

        base = programs[node]["weight"]
        res = base
        for child in programs[node]["children"]:
            res += determine_total_weight(child)

        return res
    return determine_total_weight


def find_corrected_weight(programs):
    bottom = find_bottom(programs)
    tw = totals(programs)

    def balanced(node_):
        res = len({tw(c) for c in programs[node_]["children"]}) <= 1
        return res

    for node in iterate_nodes(programs, start=bottom):
        # Node is unbalanced if it has children with different total weights
        if not balanced(node):
            # Determine which child has the wrong weight
            children = programs[node]["children"]
            child_weights = [tw(c) for c in children]
            assert len(set(child_weights)) == 2
            target_weight = max(Counter(child_weights).items(), key=lambda t: t[1])[0]
            odd_child = [c for c in children if tw(c) != target_weight][0]
            # If the 'old child' is not balanced, look deeper
            if not balanced(odd_child):
                continue

            # Determine the correct base weight to use instead
            diff = target_weight - tw(odd_child)
            oldweight = programs[odd_child]["weight"]
            corrected_weight = oldweight + diff
            return corrected_weight
        #
    #


def solve(data: str) -> tuple[int|str, int|str]:
    programs = parse(data)

    star1 = find_bottom(programs)
    print(f"Solution to part 1: {star1}")

    star2 = find_corrected_weight(programs)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2017, 7
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()