# .`··` •.· ·*    .· •·`·   . ·. · `    ·`·  .   ·· *`.   ·   `+· `·* .  ·  ·`·.
# ·*`. ··      ·`· *·.`·•·   `  . ·· Aplenty `·• * ·` ·.   · .   ·* `+·   ··` .·
# `·.  *·` · · *.`··+  https://adventofcode.com/2023/day/19 .·· *   ·*  ·`  .·+`
# +`  ·.`·.* `·.    `·    · ·  ·`   ` ··*.       `·* .`··  `  *·· ·. ·`     *·`·


import re
import typing as t
from dataclasses import dataclass

# Define aliases for some type we'll use a bit, and define type guards for some
xmastype: t.TypeAlias = t.Literal["x", "m", "a", "s"]
parttype: t.TypeAlias = dict[xmastype, int]
comparisontype: t.TypeAlias = t.Literal["<", ">"]
ruletype: t.TypeAlias = tuple[xmastype, comparisontype, int, str]
verdicttype: t.TypeAlias = t.Literal["A", "R"]
intervaltype: t.TypeAlias = dict[xmastype, tuple[int, int]]


def _is_xmas(s: str) -> t.TypeGuard[xmastype]:
    return s in t.get_args(xmastype)


def _is_comparison(s: str) -> t.TypeGuard[comparisontype]:
    return s in t.get_args(comparisontype)


def _is_verdict(s: str) -> t.TypeGuard[verdicttype]:
    return s in t.get_args(verdicttype)


def _parse_part(s: str) -> parttype:
    """Parses a line into a dictionary representing a part (with the xmas attributes)"""
    d = dict()
    items = (elem.split("=") for elem in s[1:-1].split(","))
    for k, v in items:
        assert _is_xmas(k)
        d[k] = int(v)
    
    return d


@dataclass
class Rule:
    """Represents a single 'if-then' rule, like
    if x > 2000, go to 'abc'
    Instances of this class are callable, and convert a part into a string
    representing the workflow to which the part should be moved.
    If the rule is not applicable, i.e. if the 'if' part is not satisfied,
    returns None instead."""
    
    attribute: xmastype
    comp: comparisontype
    threshold: int
    output: str

    @classmethod
    def from_string(cls, s: str) -> t.Self:
        """Helper method for constructing a rule from the format in the puzzle input"""

        m = re.match(r"(\w)([<>=])(\d+):(\w+)", s)
        assert m
        a, comp, thr, out = m.groups()
        assert _is_xmas(a) and _is_comparison(comp)
        inst = cls(attribute=a, comp=comp, threshold=int(thr), output=out)

        return inst
    
    def rule_applies(self, val: int) -> bool:
        """Determines whether the rule applies to the given value (assuming the
        value represents the relevant attribute)"""
        match self.comp:
            case ">":
                return val > self.threshold
            case "<":
                return val < self.threshold
            case _:
                raise TypeError
            #
        #
    
    def partition_interval(self, interval: tuple[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Partitions the input interval into two sub/intervals.
        The first represents the valid region, wuch that numbers within that region all
        fall inside the region to which this rule applies. The other interval is the remainder."""
        
        limits_valid = tuple(self.rule_applies(value) for value in interval)
        if all(limits_valid):
            return interval, ()
        elif not any(limits_valid):
            return (), interval
        
        low, high = interval
        match self.comp:
            case "<":
                return (low, self.threshold-1), (self.threshold, high)
            case ">":
                return (self.threshold+1, high), (low, self.threshold)
            case _:
                raise TypeError
            #
        #
    
    def __call__(self, part: parttype) -> str|None:
        """Attempt to apply the rule.
        Returns the result if the rule applies to the part, and None if not."""

        val = part[self.attribute]
        
        if self.rule_applies(val):
            return self.output
        else:
            return None
        #
    #


class Workflow:
    """Represents one of the workflows used to accept/reject machine parts.
    Contains a series of if-then rules, and an 'else' rule, where parts which
    satisfy none of the rule criteria are moved."""
    
    def __init__(self, name: str, rules: t.Iterable[Rule], else_: str) -> None:
        self.name = name
        self.else_ = else_
        self.rules = list(rules)

    def __call__(self, part: parttype) -> str:
        """Apply the workflow to the part. Returns a string representing
        the workflow (or accept/reject - "A"/"R") to which the part is moved."""
        
        for rule in self.rules:
            output = rule(part)
            if output is not None:
                return output
            #
        
        return self.else_
    #


def parse(s: str) -> tuple[list[Workflow], list[parttype]]:
    rulepart, partpart = s.split("\n\n")
    
    parts: list[parttype] = [_parse_part(partline) for partline in partpart.splitlines()]
    
    workflows: list[Workflow] = []
    for line in rulepart.splitlines():
        cut = line.index("{")
        name = line[:cut]
        conds = line[cut+1:-1].split(",")
        else_ = conds.pop()
        rules = [Rule.from_string(elem) for elem in conds]
        wf = Workflow(name=name, rules=rules, else_=else_)
        workflows.append(wf)

    return workflows, parts


class Evaluator:
    """Helper class for evaluating a machine part, using all the available workflows."""
    
    def __init__(self, workflows: t.Iterable[Workflow]) -> None:
        self._workflows: dict[str, Workflow] = dict()
        for wf in workflows:
            self._workflows[wf.name] = wf
        #
    
    def __call__(self, part: parttype, start="in") -> str:
        """Apply to a machine part.
        Returns the final evaluation of the part (should be 'A' or 'R', or I messed up)"""
        running = start
        while not _is_verdict(running):
            wf = self._workflows[running]
            running = wf(part)
        
        return running

    def count_combinations(self, lower=1, upper=4000) -> int:
        """Counts the distinct combinations of ratings which end up in the accept state.
        Starts with the extreme possible limits for each rating category,
        then repeatedly looks at the relevant rules and considers the subspaces to which
        a given rule applies and not."""
        
        subspaces: intervaltype = {c: (lower, upper) for c in t.get_args(xmastype)}
        front: list[tuple[str, intervaltype]] = [('in', subspaces)]
        next_: list[tuple[str, intervaltype]] = []
        
        res = 0
        
        while front:
            for step, space in front:
                # Check if we're done with this subset of the space
                if step == "A":
                    # If we've reached the accept state, tally up the number of combinations
                    these_counts = 1
                    for lo, hi in space.values():
                        these_counts *= (hi - lo + 1)
                    res += these_counts
                    continue
                elif step == "R":
                    continue  # if reached the reject state, do not continue
                
                # Look up the workflow we've reached so far for this subspace
                wf = self._workflows[step]
                remains = {k: v for k, v in space.items()}
                for rule in wf.rules:
                    # Determine the subsets of the interval where the rule does/does not apply
                    attr = rule.attribute
                    interval = remains[attr]
                    valid, invalid = rule.partition_interval(interval)
                    
                    # If there's a valid region, pass it to the next iteration
                    if len(valid) == 2:
                        continues = {k: valid if k == attr else v for k, v in remains.items()}
                        next_.append((rule.output, continues))
                    else:
                        continue
                    
                    # If there's an invalid region, proceed to next rule to further narrow down the space
                    assert len(invalid) == 2
                    remains = {k: invalid if k == attr else v for k, v in remains.items()}
                
                # move remaining space to the 'else' rule
                next_.append((wf.else_, remains))
            
            front = next_
            next_ = []
        
        return res


def compute_rating(*parts: parttype) -> int:
    """Computes the rating number for a machine part"""
    res = sum(v for part in parts for v in part.values())
    return res


def solve(data: str) -> tuple[int|str, ...]:
    workflows, parts = parse(data)
    
    e = Evaluator(workflows)
    
    star1 = compute_rating(*(part for part in parts if e(part) == "A"))
    print(f"Solution to part 1: {star1}")

    
    star2 = e.count_combinations()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 19
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
