# *· `·.     `.*··   · *` ·  .·*`  ·  +.  · *·. +.· ·`*`·  .  · .· *` ·  ··  *`·
# .*`·*   .  ·*     ·` · .*·   Not Enough Minerals • .· *··+  * ·.`  ·.·* + ··*.
# ·`.*.···*   `  . ·`* https://adventofcode.com/2022/day/19      `· •.* ·   +`··
#  .·  *` ·.·* · *.+  ·`   * · ·  . ` ·   . +* · `   ·.·     *·.`  · *     ·`·.*

from dataclasses import dataclass
from numba import njit
import numpy as np
from numpy.typing import NDArray
import re


raw = """Blueprint 1: Each ore robot costs 4 ore. Each clay robot costs 2 ore. Each obsidian robot costs 3 ore and 14 clay. Each geode robot costs 2 ore and 7 obsidian.
Blueprint 2: Each ore robot costs 2 ore. Each clay robot costs 3 ore. Each obsidian robot costs 3 ore and 8 clay. Each geode robot costs 3 ore and 12 obsidian."""


class Counter:
    """For keeping track of indices for array representations of states"""
    def __init__(self) -> None:
        self._running = 0
    
    @property
    def size(self) -> int:
        return self._running
    
    def __call__(self) -> int:
        res = self._running
        self._running += 1
        return res


# Indices for minerals, robots, and time left
indscounter = Counter()
minerals_order = np.array([indscounter() for _ in range(4)])
N_ORE, N_CLAY, N_OBSIDIAN, N_GEODES = minerals_order
robots_order = np.array([indscounter() for _ in range(4)])
N_ORE_ROBOTS, N_CLAY_ROBOTS, N_OBSIDIAN_ROBOTS, N_GEODE_ROBOTS = robots_order
MINUTES_LEFT = indscounter()

mineral_inds: dict[str, int] = dict(zip(('ore', 'clay', 'obsidian', 'geode'), minerals_order))
n_minerals = len(mineral_inds)

dtype = np.int64


@dataclass
class Blueprint:
    id_: int
    costs: NDArray[np.int_]


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


def _initial_state(n_minutes_total: int) -> NDArray[np.int_]:
    """Makes an array representing an initial state, with 1 ore robot, and all minutes left"""
    res = np.zeros(shape=(indscounter.size,), dtype=dtype)
    res[MINUTES_LEFT] = n_minutes_total
    res[N_ORE_ROBOTS] = 1
    return res


@njit(cache=True)
def bfs_step(
        states: NDArray[np.int_],
        costs: NDArray[np.int_],
        ceiling: NDArray[np.int_]
    ) -> NDArray[np.int_]:

    for i in range(len(states)):
        robot_type = i % n_minerals
        cost = costs[robot_type]

        available = states[i, minerals_order]
        robots = states[i, robots_order]

        # Don't buy robot if production already meets the max price for that resource
        robot_is_pointless = robots[robot_type] >= ceiling[robot_type] and robot_type != N_GEODES
        if robot_is_pointless:
            states[i, MINUTES_LEFT] = -1
            continue

        need = cost - available
        ind_need = np.where(need > 0)
        
        # There's no way to wait and buy the robot if we're short on minerals which aren't being produced
        impossible = np.any(robots[ind_need] == 0)

        if impossible:
            states[i, MINUTES_LEFT] = -1
            continue
            
        prod_time = np.ceil(need[ind_need] / robots[ind_need])
        n_turns_wait = 1 if len(prod_time) == 0 else int(prod_time.max()) + 1

        new_minerals = available + n_turns_wait*robots - cost
        states[i, minerals_order] = new_minerals
        states[i, robots_order[robot_type]] += 1
        states[i, MINUTES_LEFT] -= n_turns_wait

    return states[states[:, MINUTES_LEFT] >= 0]


def pareto_frontier(arr: NDArray[np.int_]) -> NDArray[np.bool]:
    mask = np.array([True for _ in range(len(arr))])

    for i in range(len(arr)):
        vec = arr[i]
        remaining = arr[i+1:]
        better = np.all(remaining >= vec, axis=1)
        if np.any(better):
            mask[i] = False

    return mask


def max_geodes(blueprint: Blueprint, n_minutes=24, max_iter=-1) -> int:
    record = 0

    states = np.vstack([_initial_state(n_minutes)])
    ceiling = blueprint.costs.max(axis=0)
    nits = 0

    while len(states) > 0:
        nits += 1
        if max_iter != -1 and nits > max_iter:
            raise RuntimeError

        print(len(states), record, states[:, MINUTES_LEFT].mean())
        frontier = pareto_frontier(states)
        states = states[frontier].copy()

        new_states = np.repeat(states, repeats=n_minerals, axis=0)
        new_states = bfs_step(new_states, costs=blueprint.costs, ceiling=ceiling)
        if len(new_states) == 0:
            break
        
        interpolations = new_states[:, N_GEODES] + new_states[:, N_GEODE_ROBOTS]*new_states[:, MINUTES_LEFT]
        best_interpolation_ind = np.argmax(interpolations)
        best_interpolation = interpolations[best_interpolation_ind]
        if best_interpolation > record:
            print("BEST", best_interpolation, new_states[best_interpolation_ind])
            record = best_interpolation

        # TODO TIGHTER HEURISTIC
        upper_bound = (
            new_states[:, N_GEODES]
            + new_states[:, N_GEODE_ROBOTS]*new_states[:, MINUTES_LEFT]
            + (new_states[:, MINUTES_LEFT]*(new_states[:, MINUTES_LEFT] - 1)) // 2
        )

        states = new_states[upper_bound >= record].copy()
        

    return record


def solve(data: str) -> tuple[int|str, ...]:
    blueprints = parse(data)
    
    star1 = 0
    for bp in blueprints:
        print(f"*** BLUEPRINT {bp.id_} ***")
        n_max = max_geodes(bp)
        star1 += bp.id_*n_max
    

    print(f"Solution to part 1: {star1}")

    star2 = 1
    for bp in blueprints[:3]:
        star2 *= max_geodes(bp, n_minutes=32)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 19
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
