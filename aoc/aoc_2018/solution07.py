# .•⸳ꞏ*   ꞏ. `* ꞏ`.ꞏ  ⸳`ꞏ.* . ⸳• + ꞏ⸳* . ` .ꞏ  +   ⸳. ꞏ+ `ꞏ `     .` *ꞏ⸳ ꞏ⸳  .* 
#  ꞏ.`  ꞏ   *ꞏ.    +ꞏ . ⸳ `⸳ꞏ  The Sum of Its Parts ꞏ.` ⸳ *  ꞏꞏ *    ⸳  +` .  ꞏ•
# ꞏ⸳  `⸳ .• `    ꞏ ⸳•  https://adventofcode.com/2018/day/7 ꞏ+   . ⸳   *ꞏ`⸳. +ꞏ.`
# ⸳+ .  ``     ꞏ.*  ꞏ⸳     *ꞏ⸳ .    * ` ꞏ.    `⸳ꞏ* .    •ꞏ`*⸳ .  ꞏ   . `⸳ꞏ ⸳`* .


import re
import string
from typing import Iterator, TypeAlias

graph_type: TypeAlias = dict[str, set[str]]
 
 
test = """Step C must be finished before step A can begin.
Step C must be finished before step F can begin.
Step A must be finished before step B can begin.
Step A must be finished before step D can begin.
Step B must be finished before step E can begin.
Step D must be finished before step E can begin.
Step F must be finished before step E can begin."""
 
 
def parse(s):
    res = []
    p = r"Step (\S) must be finished before step (\S) can begin."
    for line in s.splitlines():
        res.append(re.match(p, line).groups())
    return res


def build_dag(edges: list[tuple[str, str]]) -> graph_type:
    """Constructs a simple DAG, represented as dict where nodes map to a set of their neighbors."""
    
    nodes = sorted(set(sum(map(list, edges), [])))
    G: graph_type = {u: set([]) for u in nodes}
    for u, v in edges:
        G[u].add(v)
    
    return G


def traverse_dag(G: graph_type, workloads: dict[str, int], n_workers: int=1) -> Iterator[tuple[int, str]]:
    """Traverses the DAG and generates tuples of elapsed time, char, denoting the total time elapsed at the time when
    each character is completed.
    workloads is a dict representing the time in seconds required to complete each task (represented by the chars).
    n_workers is the number of worker elves that may be concurrently allocated to aolving a task."""
    
    # Map each task to a set of 'required' tasks which must be solved before beginning the task
    requirements: dict[str, set[str]] = {u: set([]) for u in G.keys()}
    for k, neighbors in G.items():
        for v in neighbors:
            requirements[v].add(k)
        #

    active: dict[str, int] = dict()  # maps each active task to the time remaining
    finished: set[str] = set([])
    elapsed = 0
    
    def get_available(n) -> Iterator[str]:
        """Generates n available tasks that we may begin now.
        Returns unfinished tasks that are not currently being worked on, and which
        have all their requirements satisfied."""
        
        returned = 0
        for task, reqs in sorted(requirements.items()):
            free = task not in finished and task not in active
            unlocked = reqs.issubset(finished)
            if free and unlocked and returned < n:
                yield task
                returned += 1
    
    while finished != set(G.keys()):
        # Start work on available tasks
        n_free_workers = n_workers - len(active)
        for task in get_available(n_free_workers):
            active[task] = workloads[task]
        
        # Update progress on task. Use the least remaining amount of work as step size
        tick_size = min(active.values())
        elapsed += tick_size
        active = {k: v - tick_size for k, v in active.items()}
        done = sorted(k for k, v in active.items() if v == 0)
        
        # Update data on active and finished tasks
        for char in done:
            del active[char]
            finished.add(char)
            yield elapsed, char
        

def solve(data: str, delay: int=60, n_workers: int=5):
    edges = parse(data)
    
    G = build_dag(edges)
    
    # Build the string made by taking the task letters in the order of completion
    workloads = {char: 1 for char in string.ascii_uppercase}
    star1 = "".join((s for _, s in traverse_dag(G, workloads=workloads, n_workers=1)))
    print(f"Solution to part 1: {star1}")
    
    # For the second problem, we need the amount of time required, so just take the final time
    workloads2 = {char: 1 + delay + i for i, char in enumerate(string.ascii_uppercase)}
    star2 = list(traverse_dag(G=G, n_workers=n_workers, workloads=workloads2))[-1][0]
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2018, 7
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve, extra_kwargs_parser=lambda d: d)
    from aocd import get_data
    
    
    raw = get_data(year=year, day=day)
    #raw = test
    # 947 too low!!!
    solve(raw)


if __name__ == '__main__':
    main()
