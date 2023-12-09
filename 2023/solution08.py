import math


def read_input():
    with open("input08.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    instructions, nodepart = s.split("\n\n")

    nodes = {}
    for line in nodepart.split("\n"):
        source, neighbortuple_s = line.split(" = ")
        neighbortuple = tuple(neighbortuple_s[1:-1].split(", "))
        nodes[source] = neighbortuple

    return instructions, nodes


def walk(instructions, nodes, startnode):
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


def follow_instructions(instructions, nodes):
    n_steps = 0
    destination_node = "ZZZ"
    for node in walk(instructions, nodes, startnode="AAA"):
        if node == destination_node:
            break
        n_steps += 1

    return n_steps


def trace_cycle(instructions, nodes, startnode):
    """Takes a starting node and returns the index at which a cycle begins,
    the index *within the cycle* at which there's a destination (Z) node,
    and the cycle length."""

    seen = set([])
    path = []
    n_steps = 0
    cycle_start = None

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

    cycle_len = n_steps - cycle_start
    hits = [i for i, n in enumerate(path) if n[-1] == "Z"]
    assert len(hits) == 1
    hit = hits[0] - cycle_start

    return cycle_start, hit, cycle_len


def count_ghost_steps(instructions, nodes):
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


def main():
    raw = read_input()
    instructions, nodes = parse(raw)

    star1 = follow_instructions(instructions, nodes)
    print(f"Arrival after {star1} steps.")

    star2 = count_ghost_steps(instructions, nodes)
    print(f"All ghosts arrive at their destinations after {star2} steps.")


if __name__ == '__main__':
    main()
