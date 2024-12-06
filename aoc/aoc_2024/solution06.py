#  . ⸳ꞏ  *+.`    ꞏ   .` ⸳     ꞏ ꞏ `⸳* .  ⸳  *` ꞏ ` +` ꞏ  .  ⸳+ꞏ .     .•`⸳*ꞏ   ⸳
# ⸳`ꞏ   ⸳.`   ⸳•ꞏ`* `   .     ⸳+ Guard Gallivant   ⸳+* . ꞏ`     +  ⸳` *ꞏ•   ꞏ⸳ ⸳
# •⸳.` .    ꞏ .⸳ +`.   https://adventofcode.com/2024/day/6     .⸳+ `ꞏ.  ꞏ*ꞏ   •`
# ꞏ * `*  .+  •` ꞏ.ꞏ *`  *     ꞏ+ ⸳  +  `.*⸳    * ⸳` . + `ꞏ`* . + `  ꞏ  ⸳•⸳  ꞏ`•


from collections import defaultdict
import numpy as np


def parse(s):
    m = np.array([list(line) for line in s.splitlines()])
    return m


#Stuff for rotating direction vectors right
dirs = ((-1, 0), (0, 1), (1, 0), (0, -1))
rot_dict = {dirs[i]: dirs[(i+1) % len(dirs)] for i in range(len(dirs))}


def get_initial_pos(map_: np.ndarray) -> tuple:
    """Grabs initial position from the array"""
    for pos in np.ndindex(map_.shape):
        if map_[pos] == "^":
            return pos
        #
    #

def _add_tuples(a, b):
    res = tuple(x + y for x, y in zip(a, b, strict=True))
    return res


class Graph:
    """Determines on initializaion which states (position, direction) map to which subsequent states."""

    def __init__(self, map_: np.ndarray):
        self.edges = dict()
        
        # Reverse map to make it easy to look up which states lead to a given state
        self._reverse = defaultdict(lambda: set([]))
        
        # Go over all adjacent states
        for pos in np.ndindex(map_.shape):
            for dir_ in dirs:
                
                u = (pos, dir_)
                
                # Check if taking a step forward results in falling off the map
                forward = _add_tuples(pos, dir_)
                inside_map = all(0 <= crd < dim for crd, dim in zip(forward, map_.shape, strict=True))
                if not inside_map:
                    continue
                
                # The subsequent state is a step forward, or a right turn in case of an obstacle
                if map_[forward] == "#":
                    v = (pos, rot_dict[dir_])
                else:
                    v = (forward, dir_)
                
                self.edges[u] = v
                self._reverse[v] |= {u}
            #
        #
    
    def _redirect(self, blocked: tuple):
        """Generates a dict of the state transitions affected by inserting an obstacle at the specified location.
        The dict containts states as keys and 'corrected' subsequent states as values, and so can be used to override
        the graph class' edges to simulate traversal with an extra obstacle."""

        res = dict()
        
        # Go over the 4 states that might occur at the input position (one for each direction)
        for dir_ in dirs:
            state = (blocked, dir_)
            for neighbor in self._reverse[state]:
                # The neighbors pointing to the blocked site whould instead turn right
                npos, ndir = neighbor
                ndir = rot_dict[ndir]
                redirected = (npos, ndir)
                
                assert neighbor not in res
                res[neighbor] = redirected
            #
        
        return res

    def traverse(self, initial_state: tuple, blocked: tuple=None):
        """Traverses the graph starting from the specified state.
        if a blocked coordinate is provided, the traversal will simulate an obstacle there."""

        state = initial_state
        
        # Transitions affected by the new obstacle        
        redir = self._redirect(blocked=blocked) if blocked else dict()
        
        while True:
            yield state
            
            # Check if transition is affected by an added obstacle
            try:
                state = redir[state]
                continue
            except KeyError:
                pass
            
            # Otherwise follow the normal transitions
            try:
                state = self.edges[state]
            except KeyError:
                break  # If there's no next state, the route goes off the map, so stop iterating
            #
        #
    
    def state_leads_to_loop(self, state: tuple, insert_obstacle_at: tuple):
        """Checks if inserting an obstacle at the specified location results in a loop starting from the given state."""
        
        # Maintain set of all states encountered
        history = set([])
        
        # There's a loop if we arrive at a previously encountered state
        for other in self.traverse(initial_state=state, blocked=insert_obstacle_at):
            if other in history:
                return True
            history.add(other)
        
        return False


def compute_patrol_length(G: Graph, initial_state: tuple):
    """Computes the number of locations visited by the guard"""
    visited = {pos for pos, _ in G.traverse(initial_state=initial_state)}
    res = len(visited)    
    return res


def count_potential_loops(G: Graph, initial_state: tuple):
    """Determines the number of sites where an obstacle can be inserted to introduce a cycle in the guard's patrol"""
    res = 0
    default_path = list(G.traverse(initial_state=initial_state))
    
    # Exclude the initial position from obstacles
    initial_pos, _ = initial_state
    forbidden_sites = {initial_pos}
    
    # Go over all pairs of subsequent states, and try inserting an obstacle at the latter's position
    for i in range(len(default_path) - 1):
        state = default_path[i]
        pos, _ = state
        next_pos, _ = default_path[i+1]
        
        # Only insert if the position changes between the two states, and if the position is not 'forbidden'
        try_insert = next_pos != pos and next_pos not in forbidden_sites
        if try_insert:
            loop = G.state_leads_to_loop(state=state, insert_obstacle_at=next_pos)
            res += loop
            
            # Add to forbidden set to avoid meaningless results (e.g. cycle after a blocked path)
            forbidden_sites.add(next_pos)
            
    
    return res



def solve(data: str):
    m = parse(data)
    
    G = Graph(m)
    initial_state = (get_initial_pos(m), (-1, 0))
    star1 = compute_patrol_length(G=G, initial_state=initial_state)

    print(f"Solution to part 1: {star1}")

    star2 = count_potential_loops(G=G, initial_state=initial_state)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2024, 6
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
