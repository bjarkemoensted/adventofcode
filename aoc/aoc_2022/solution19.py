# *· `·.     `.*··   · *` ·  .·*`  ·  +.  · *·. +.· ·`*`·  .  · .· *` ·  ··  *`·
# .*`·*   .  ·*     ·` · .*·   Not Enough Minerals • .· *··+  * ·.`  ·.·* + ··*.
# ·`.*.···*   `  . ·`* https://adventofcode.com/2022/day/19      `· •.* ·   +`··
#  .·  *` ·.·* · *.+  ·`   * · ·  . ` ·   . +* · `   ·.·     *·.`  · *     ·`·.*

import re
from dataclasses import dataclass
from enum import IntEnum
from functools import partial, reduce
from operator import mul

import numpy as np
from numpy.typing import NDArray


class Inds(IntEnum):
    """Indices for number of minerals and robots, and time left in a state"""
    N_ORE = 0
    N_CLAY = 1
    N_OBSIDIAN = 2
    N_GEODES = 3
    N_ORE_ROBOTS = 4
    N_CLAY_ROBOTS = 5
    N_OBSIDIAN_ROBOTS= 6
    N_GEODE_ROBOTS = 7
    MINUTES_LEFT = 8


minerals_order = np.array([Inds.N_ORE, Inds.N_CLAY, Inds.N_OBSIDIAN, Inds.N_GEODES])
robots_order = np.array([Inds.N_ORE_ROBOTS, Inds.N_CLAY_ROBOTS, Inds.N_OBSIDIAN_ROBOTS, Inds.N_GEODE_ROBOTS])
mineral_inds: dict[str, int] = dict(zip(('ore', 'clay', 'obsidian', 'geode'), minerals_order))
n_minerals = len(mineral_inds)

dtype = np.int32
type arrtype = NDArray[dtype]


@dataclass
class Blueprint:
    id_: int
    costs: arrtype


def parse(s: str) -> list[Blueprint]:
    res: list[Blueprint] = []
    for line in s.splitlines():
        arr = np.zeros(shape=(len(minerals_order), len(minerals_order)), dtype=dtype)
        a, b = line.split(": ")
        id_ = int(a.split("Blueprint ")[-1])
        for part in b.split(". "):
            m = re.match("Each (.*) robot costs (.*)", part.strip())
            assert m
            robot_type, price_string = m.groups()
            robot_ind = mineral_inds[robot_type]
            for n_str, mineral_type in map(str.split, price_string.replace(".", "").split(" and ")):
                cost_ind = mineral_inds[mineral_type]
                arr[robot_ind, cost_ind] = int(n_str)
            #
        res.append(Blueprint(id_=id_, costs=arr))

    return res


def bfs_step(
        states: arrtype,
        costs: arrtype
    ) -> arrtype:
    """Takes a 2D array where each row represents a state (counts of each mineral and robot type,
    and number of minutes left). Returns a new 2D array representing the possible next states,
    after attempting to save up to each possible robot type."""

    # Make N copies of current states (one for each robot type to buy)
    new_states = np.repeat(states, repeats=n_minerals, axis=0)
    nrows, _ = new_states.shape
    inds = np.arange(nrows)
    robot_types = inds % n_minerals  # which robot to buy from each state

    # How much more we need of each mineral type
    prices = costs[robot_types]
    available = new_states[:, minerals_order]
    needed = prices - available
    # Upper bound on each robot type
    ceiling = costs.max(axis=0)

    # Determine the time until we've saved enough for each robot type
    robots = new_states[:, robots_order]
    # Default to infinity, then compute values for owned robot types, to avoid zero division
    production_times = np.full(shape=robots.shape, fill_value=np.inf)
    producible_mask = (needed > 0) & (robots > 0)
    production_times[producible_mask] = needed[producible_mask] / robots[producible_mask]
    # No time is needed when we already have sufficient minerals
    production_times[needed <= 0] = 0
    prod_time = np.ceil(production_times.max(axis=1))

    # Set minutes left to a negative number where required minerals aren't produced
    impossible_mask = np.isinf(prod_time)
    new_states[impossible_mask, Inds.MINUTES_LEFT] = -1

    # Wait for one additional minute, so the new robot is ready
    possible_mask = ~impossible_mask
    wait_times = prod_time[possible_mask].astype(int) + 1

    # Update resources and remaining time
    produced = wait_times[:, None]*robots[possible_mask]
    updated_minerals = (produced - prices[possible_mask])
    new_states[np.ix_(possible_mask, minerals_order)] += updated_minerals
    new_states[possible_mask, Inds.MINUTES_LEFT] -= wait_times
    # Increment number of robots
    robot_inds = robots_order[robot_types]
    new_states[inds, robot_inds] += 1

    # Look for 'pointless' states where a mineral production exceeds max daily needs
    robot_checks = robots_order[robots_order != Inds.N_GEODE_ROBOTS]
    pointless_mask = np.any(new_states[:, robot_checks] > ceiling[minerals_order != Inds.N_GEODES], axis=1)
    new_states[pointless_mask, Inds.MINUTES_LEFT] = -1
    
    # Return feasible states
    res = new_states[new_states[:, Inds.MINUTES_LEFT] > 0]
    return res


def make_initial_state(n_minutes_total: int) -> arrtype:
    """Makes an array representing an initial state, with 1 ore robot, and all minutes left"""
    arr = np.zeros(shape=(len(Inds),), dtype=dtype)
    arr[Inds.MINUTES_LEFT] = n_minutes_total
    arr[Inds.N_ORE_ROBOTS] = 1

    res = np.vstack([arr])
    return res


def heuristic(states: arrtype) -> arrtype:
    """Computes an upper bound on the number of geodes that can be attained in the remaining time.
    Works by assuming a free geode robot can be produced every turn.
    The geodes produced by these free robots amount to
    0 + 1 + ... + n-1 = n(n-1)/2.
    This number is added to the production from the current robots to achieve the bound."""
    
    res = (
        states[:, Inds.N_GEODES]
        + states[:, Inds.N_GEODE_ROBOTS]*states[:, Inds.MINUTES_LEFT]
        + (states[:, Inds.MINUTES_LEFT]*(states[:, Inds.MINUTES_LEFT] - 1)) // 2
    )
    return res


def max_geodes(blueprint: Blueprint, n_minutes=24) -> int:
    """Computes the maximum possible number of geodes attainable with the input blueprint,
    in the input amount of turns.
    Starts with the initial state, then repeatedly expands current states by attempting to
    buy each of the 4 robot types from each state, waiting the necessary amount of turns until
    sufficient minerals are available. States are truncated by maintaining the current greatest amount,
    discarding states with a lower upper bound."""
    
    record = 0
    states = make_initial_state(n_minutes_total=n_minutes)

    while len(states) > 0:
        # Update states
        new_states = bfs_step(states, costs=blueprint.costs)
        
        # Check if any exceed the current record for number of geodes
        interpolations = (
            new_states[:, Inds.N_GEODES]
            + new_states[:, Inds.N_GEODE_ROBOTS]*new_states[:, Inds.MINUTES_LEFT]
        )
        best_interpolation = np.max(interpolations, initial=0)
        if best_interpolation > record:
            record = int(best_interpolation)

        # Drop states which can never exceed the current best
        upper_bound = heuristic(new_states)
        states = new_states[upper_bound >= record]
        

    return record


def quality_level(blueprint: Blueprint, n_minutes=24) -> int:
    """Computes the quality level of a blueprint, by multiplying its ID with the maximum
    number of geodes attainable in the available time."""

    n_max = max_geodes(blueprint=blueprint, n_minutes=n_minutes)
    res = blueprint.id_ * n_max
    return res


def solve(data: str) -> tuple[int|str, ...]:
    blueprints = parse(data)
    
    star1 = sum(map(quality_level, blueprints))
    print(f"Solution to part 1: {star1}")

    geodes_extra_time = partial(max_geodes, n_minutes=32)
    star2 = reduce(mul, map(geodes_extra_time, blueprints[:3]), 1)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 19
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
