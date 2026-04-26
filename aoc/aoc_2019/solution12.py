# ..·  · *   .  · `  . *+`·   *·.·+   ·.  *    `*. +·· *.· ` *.·    ` ·  .. ·*·`
# ·*`·.. · ·*`  .  ·   `·. * ·  The N-Body Problem ·* .`  ·   ·+ .·     *·`* .`·
# ·`.•  ·  . *· •  .   https://adventofcode.com/2019/day/12 .· `  *  .  .  ·+`·.
# *·.`· *+  · .  . *·`  .* · . · . *·` ·.*  . *···. `  .·* ·  *.`  ·• . · * . .·

import math
from numba import njit
import numpy as np
from numpy.typing import NDArray
import re
from typing import Self


def parse(s: str) -> NDArray[np.int_]:
    vals = []
    for line in s.splitlines():
        m = re.match(r"<x=(-?\d+), y=(-?\d+), z=(-?\d+)>", line)
        assert m is not None
        vals.append(list(map(int, m.groups())))
    
    res = np.array(vals, dtype=int)
    return res


@njit
def _update_state(state: NDArray[np.int_]) -> None:
    """Update the state of the moon system. This is just to run updates in an optimized numba function"""
    
    _, n, dim = state.shape

    # Update velocities due to gravity
    for d in range(dim):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Find coordinate difference between every pair of moons
                diff = state[0, j, d] - state[0, i, d]
                if diff > 1:
                    diff = 1
                elif diff < -1:
                    diff = -1
                
                # Add difference (capped to [-1, +1]) to velocity
                state[1, i, d] += diff
            #
        #
    
    # Update positions as well
    for i in range(n):
        for d in range(dim):
            state[0, i, d] = state[0, i, d] + state[1, i, d]


class Simulation:
    def __init__(self, pos: NDArray[np.int_]) -> None:
        self.state = np.stack([pos.copy(), np.zeros_like(pos)])
        _, self.n_moons, self.dim = self.state.shape
        self.n_time_steps_elapsed = 0
        
        # Store a hash for the initial state in the x, y, and z directions, independently
        self._initial_keys = [self._key(self.state[:, :, i]) for i in range(self.dim)]
        self._recurrence_times = np.full(shape=(self.dim,), fill_value=-1, dtype=int)
         
    def _key(self, arr: NDArray[np.int_]) -> bytes:
        """Provides a hash for the state at a given time"""
        return np.ascontiguousarray(arr).tobytes()

    def tick(self, n=1) -> Self:
        """Simulates one or more time steps"""
        for _ in range(n):
            _update_state(self.state)
            self.n_time_steps_elapsed += 1
            self._record()
        
        return self
    
    def _record(self) -> None:
        """Records any new recurrence times determined"""
        for i in range(self.dim):
            if self._recurrence_times[i] != -1:
                continue
            
            # If the hash along this dimention has returned to initial val, we've found its recurrence
            key = self._key(self.state[:, :, i])
            if key == self._initial_keys[i]:
                self._recurrence_times[i] = self.n_time_steps_elapsed
            #
        #

    def run_until_recurrence(self) -> Self:
        """Runs the simulation until the recurrence time in each dimension has been determined"""
        while np.any(self._recurrence_times == -1):
            self.tick()
        return self

    def total_energy(self) -> int:
        """Computes the 'total energy' of the moon system"""
        pos, vel = self.state
        # Potential + kinetic energy for each moon
        e_pot = np.abs(pos).sum(axis=1)
        e_kin = np.abs(vel).sum(axis=1)
        # Compute total energy for each moon, and sum them
        e_tot = e_pot*e_kin
        res = e_tot.sum().item()
        return res

    def recurrence_time(self) -> int:
        """Returns the recurrence time for the full system. Recurrence times for each spatial
        dimension must be found prior to calling this (i.e. the .run_until_recurrence method)"""
        
        if np.any(self._recurrence_times == -1):
            raise ValueError(f"Simulation must run until we have recurrence times for each dimension")

        # Recurrence time for the system is the least common multiple of the independent times for each dim
        res = math.lcm(*self._recurrence_times)
        return res


def solve(data: str) -> tuple[int|str, ...]:
    arr = parse(data)

    star1 = Simulation(arr).tick(1_000).total_energy()
    print(f"Solution to part 1: {star1}")

    star2 = Simulation(arr).run_until_recurrence().recurrence_time()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 12
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
