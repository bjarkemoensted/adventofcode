# *·.`  * ·`  ·   . `  ·  +`  ·  *·`. `   · ` . `·+  ·*    ·`.+ * · `·• .·  *·`.
# .`*· · ·. `*   ` ·  `.·  •· . Haunted Wasteland  ·  ·`* ` ·   ·  ` *·  `  `..·
# ·.`*· . * +     ·`·  https://adventofcode.com/2023/day/8 .    `·    `· .  . ·`
# ` ·• .`  ··`*.`·   · ` * · . ` . ·*`·  · `.* ·  ·`*. ·`.  `·.· *`  ·   *· `  ·

import math
from typing import TypeAlias


nodemap: TypeAlias = dict[str, tuple[str, str]]


def parse(s: str) -> tuple[str, nodemap]:
    """Extracts the instructions (string of chars) and nodes (dict
    mapping strings to tuple of the left/right nodes)."""

    instructions, nodepart = s.split("\n\n")

    nodes = {}
    for line in nodepart.split("\n"):
        source, neighbortuple_s = line.split(" = ")
        left, right = neighbortuple_s[1:-1].split(", ")
        nodes[source] = (left, right)

    return instructions, nodes


def walk(instructions: str, nodes: nodemap, startnode: str):
    """Starts at the startnode, then follows the instructions, given the nodes."""
    
    d = {"L": 0, "R": 1}
    instructions_ind = [d[elem] for elem in instructions]
    i = 0
    current_node = startnode

    while True:
        yield current_node

        ind = instructions_ind[i]
        neighbors = nodes[current_node]
        current_node = neighbors[ind]
        i = (i + 1) % len(instructions_ind)


def follow_instructions(instructions: str, nodes: nodemap, startnode: str="AAA", destination_node: str="ZZZ") -> int:
    """Computes the number of steps before reaching the destination"""
    n_steps = 0

    for node in walk(instructions, nodes, startnode=startnode):
        if node == destination_node:
            break
        n_steps += 1

    return n_steps


def trace_cycle(instructions: str, nodes: nodemap, startnode: str="AAA") -> tuple[int, int, int]:
    """Takes a starting node and returns the index at which a cycle begins,
    the index *within the cycle* at which there's a destination (Z) node,
    and the cycle length."""

    seen = set([])
    path: list[str] = []
    n_steps = 0
    cycle_start = -1

    for node in walk(instructions, nodes, startnode):
        ind = n_steps % len(instructions)
        state = (ind, node)
        if state in seen:
            cycle_start = next(ii for ii, n in enumerate(path) if n == node and ii % len(instructions) == ind)
            break
        else:
            seen.add(state)
            path.append(node)
            n_steps += 1
        #
    
    if cycle_start == -1:
        raise RuntimeError
    
    cycle_len = n_steps - cycle_start
    hits = [i for i, n in enumerate(path) if n[-1] == "Z"]
    assert len(hits) == 1
    hit = hits[0] - cycle_start
    
    return cycle_start, hit, cycle_len


def count_ghost_steps(instructions, nodes) -> int:
    # Grab info on cycle start, Z index, and cycle length for all the ghost paths
    start_nodes = [u for u in nodes.keys() if u[-1] == "A"]
    not_arrived = [trace_cycle(instructions, nodes, start_node) for start_node in start_nodes]

    n_steps = 0
    stepsize = 1

    while not_arrived:
        # check arrivals
        for i in range(len(not_arrived) - 1, -1, -1):
            cycle_start, hit, cycle_len = not_arrived[i]
            # The ghost has arrived if it has entered its cycle and hits a Z-node
            ghost_arrives = n_steps > cycle_start and (n_steps - cycle_start) % cycle_len == hit
            if ghost_arrives:
                del not_arrived[i]
                # This ghost will only arrive in multiples of its cycle length. Update step size to skip
                stepsize = math.lcm(stepsize, cycle_len)

        if not_arrived:
            n_steps += stepsize

    return n_steps


def solve(data: str) -> tuple[int|str, ...]:
    instructions, nodes = parse(data)

    star1 = follow_instructions(instructions, nodes)
    print(f"Solution to part 1: {star1}")

    star2 = count_ghost_steps(instructions, nodes)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 8
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
