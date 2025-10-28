# •·.`·  `+ . `*·        · ·`  .+`·    ·* .·*    · *·  . `+ · .  ·*      ·*`+.·`
# `*+··` .. · +    ·    ` ·*·* `  Repose Record   *    ·*    +`·.    *·.·   `· *
# ·*`.  · `    ` · *·  https://adventofcode.com/2018/day/4    + *  ··. `  ·*.` ·
# ·`· . +·    ·.+*` .  *·`    + ·*. · `·   *`· .     •· `·*     ·. * ` ·* `· +·`

from dataclasses import dataclass
import numpy as np
import re
from typing import cast, Iterator, Literal, TypeAlias


temporal: TypeAlias = tuple[int, ...]
event: TypeAlias = tuple[temporal, Literal[0, 1]]


@dataclass
class Shift:
    guard: int
    start: temporal
    events: list[event]


def make_shifts(pairs: list[tuple[temporal, str]]) -> Iterator[Shift]:
    """Construct a shift instance for each shift for each guard."""
    
    d = {"falls asleep": 1, "wakes up": 0}
    guard = None
    start: temporal|None = None
    events: list[event] = []
    for time_, text in pairs:
        if "#" in text:
            if guard is not None:
                yield Shift(guard=guard, start=cast(temporal, start), events=events)
            guard = int(text.split()[1][1:])
            start = time_
            events = []
        else:
            e = d[text]
            events.append(cast(event, (time_, e)))
        
        
    yield Shift(guard=cast(int, guard), start=cast(temporal, start), events=events)


def parse(s: str) -> list[Shift]:
    res = []
    
    pairs = []
    for line in s.splitlines():
        m = re.match(r'\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})\] (.*)', line)
        assert m
        time_ = tuple(map(int, m.groups()[:-1]))
        text = m.groups()[-1]
        pairs.append((time_, text))
    
    pairs.sort()
    
    res = [shift for shift in make_shifts(pairs)]
    return res


def asleep_by_minute(shifts: list[Shift]) -> dict[int, list[int]]:
    """For each guard, makes an array of counts of the number of instances where the guard
    was a asleep at the given minute"""
    
    d: dict[int, np.typing.NDArray[np.int_]] = dict()
    
    for shift in shifts:
        # Start off assuming guard is never asleep
        arr = np.array([0 for _ in range(60)])
        # For each chronological event, update remaining minutes (to sleep/wake according to event)
        for time_, e in shift.events:
            assert time_[-2] == 0
            minutes = time_[-1]
            arr[minutes:] = e
        # If this is the first shift we're seeing, use the minute array, otherwise add to existing
        try:
            d[shift.guard] += arr
        except KeyError:
            d[shift.guard] = arr
    
    res = {k: [int(elem) for elem in v] for k, v in d.items()}
    return res


def choose(d) -> int:
    """Solve riddle 1 - which guard spendsthe most time asleep"""
    guard, sleep_amount_for_minute = max(d.items(), key=lambda t: sum(t[1]))
    minute = max(range(len(sleep_amount_for_minute)), key=lambda i: sleep_amount_for_minute[i])
    return guard*minute


def guard_with_minute_record(d) -> int:
    """Solve riddle 2 - which guard at which minute is more often asleep"""
    res = -1
    record = float("-inf")
    for guard, vals in d.items():
        # Update result if the record for most times asleep on a specific minute
        maxind = max(range(len(vals)), key=vals.__getitem__)
        max_ = vals[maxind]
        if max_ > record:
            record = max_
            res = guard*maxind
    
    return res


def solve(data: str) -> tuple[int|str, int|str]:
    shifts = parse(data)
    
    d = asleep_by_minute(shifts)
    
    star1 = choose(d)
    print(f"Solution to part 1: {star1}")

    star2 = guard_with_minute_record(d)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 4
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
