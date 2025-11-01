# · *  .·`· .`*·   .*  .·*  ·+ •  .`·.    · * ` .·     · • *·. *`   ·   *·  `·.·
#   ·  · .* `    * ··  +   Subterranean Sustainability + ·   ·   .*·*`   •· .·` 
# `* ·.  · `  · *`  .· https://adventofcode.com/2018/day/12   +`*·· .    *` · .`
#  ·.•·`*+`··+ .·    +· `  ·  +. ·•*` ·  .  ·*  ·   `·   .* + `·   * ·   `.•·`·*

from collections import defaultdict
from typing import TypeAlias

statetype: TypeAlias = dict[int, str]
hashtype: TypeAlias = tuple[int, ...]


def parse(s: str) -> tuple[str, dict[str, str]]:
    a, b = s.split("\n\n")
    
    initial_state = a.split(": ")[-1]
    d = {left: right for left, right in map(lambda s: s.split(" => "), b.splitlines())}
    
    return initial_state, d


class PlantSimulator:
    def __init__(self, initial_state: str, transitions: dict[str, str]):
        self.initial_state: statetype = defaultdict(lambda: ".")
        for i, char in enumerate(initial_state):
            if char == "#":
                self.initial_state[i] = char
            
        self._registry: dict[hashtype, list[int]] = dict()
        self.history: list[statetype] = []
        self._register_state(self.initial_state)
        self.transitions = {k: v for k, v in transitions.items() if v =="#"}
        
        _trans_lens = {len(k) for k in self.transitions.keys()}
        assert len(_trans_lens) == 1
        # Make sure empty areas are inactive (would be aburd as the space is infinite, but still)
        assert not any(all(c == "." for c in s) for s in self.transitions.keys())
        self._transition_length = list(_trans_lens)[0]

    @staticmethod
    def rep_state(state: statetype, shift_first_to: int|None=0) -> hashtype:
        """Converts state into an ordered tuple of indices of plants.
        shift_first_to shifts all indices so the leftmost plan is at the provided index.
        This is useful for spotting if states are identical except a translation.
        The result of this is hashable and can be used to efficiently detect recurring states."""

        keys = sorted(k for k, v in state.items() if v == "#")
        shift = 0 if shift_first_to is None else shift_first_to - keys[0]
        res = tuple(v + shift for v in keys)
        return res

    def _register_state(self, state: statetype) -> bool:
        """Registers state. Returns bool, indicating whether state was seen before"""
        k = self.rep_state(state)
        seen_before = k in self._registry
        self.history.append(state)
        ind = len(self.history) - 1
        
        if seen_before:
            self._registry[k].append(ind)
        else:
            self._registry[k] = [ind]
        
        return seen_before

    def tick(self) -> bool:
        """Produces the next generation of plants.
        Returns a boolean indicating whether the resulting state has been seen before"""
        
        next_state = defaultdict(lambda: ".")
        old = self.history[-1]
        locs = sorted(old.keys())
        # Figure out how many spaces around each plant we need to check
        n_chars = self._transition_length
        offset = n_chars // 2
        
        for i in range(locs[0] - n_chars, locs[-1] + n_chars):
            # Represent adjacent pots and look for matching transition rules
            k = "".join(old[shift] for shift in range(i, i+n_chars))
            try:
                subs = self.transitions[k]
                next_state[i+offset] = subs
            except KeyError:
                pass  # if no rule match, let the pot remaing empty
            #
        
        return self._register_state(next_state)
    
    def code_after_n(self, n: int) -> int:
        """Returns the 'code' (sum of indices of plants) after n generations.
        Exploits recurrences by detecting 'loops' in the generations, meaning states that are
        identical up to a constant shift."""
        
        # Look for recurrences. Keep iterating until one is found, or we have the required n gens
        recurrences = [k for k, v in self._registry.items() if len(v) > 1]
        recurrence_found = len(recurrences) > 0
        if not recurrence_found:
            while not self.tick():
                if len(self.history) > n:
                    break
                #
            #
        
        # If we hit the rquired n gens without detecting a recurrence, just use the last state
        if len(self.history) > n:
            return sum(k for k, v in self.history[n].items() if v == "#")
        else:
            # Get the generations of the recurrence
            rec_inds = next(v for v in self._registry.values() if len(v) > 1)
            a, b = rec_inds[-2:]
            loop_size = b - a
            
            # Find leftmost plant before/after a loop, and the shift cause by a single loop
            state_tuples = [self.rep_state(self.history[ind], shift_first_to=None) for ind in (a, b)]
            shift = state_tuples[1][0] - state_tuples[0][0]
            n_repeats = (n - a) // loop_size
            # Compute the total shift by completing all loops
            total_shift = n_repeats*shift
            
            # Compute where in the loop we end up after completing all n generations
            remainder = n % loop_size
            final_ind_in_loop = self.history[a+remainder]
            
            return sum(k + total_shift for k, v in final_ind_in_loop.items() if v == "#")
        #
    #


def solve(data: str) -> tuple[int|str, int|str]:
    initial_state, d = parse(data)
    ps = PlantSimulator(initial_state=initial_state, transitions=d)

    star1 = ps.code_after_n(n=20)
    print(f"Solution to part 1: {star1}")
    
    n_gens = 50_000_000_000
    star2 = ps.code_after_n(n_gens)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 12
    from aocd import get_data
    
    raw = get_data(year=year, day=day)
    solve(raw)
    

if __name__ == '__main__':
    main()
