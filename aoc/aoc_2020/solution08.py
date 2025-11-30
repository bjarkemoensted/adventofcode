# ·.• .*  ·`. ·*  ·    `·*        `·  + *`··`.    · *   ·*   ` . ·   * ·.  .` *·
#  ·.   `*. · `· +. `+·   ·  `*  Handheld Halting ` · . `·+ · .`    *·.   ·` .·+
# *.·  `·`·  •+. ·` ·` https://adventofcode.com/2020/day/8 .*  ·. ` · + `. · +`·
#  `.· ·    *.· ` *  ·.   ` ·  * +·•     . `· ·+· +..     `·  · ·.     *· +  ·.`

import networkx as nx


def parse(s: str) -> list[tuple[str, int]]:
    res = []
    for line in s.splitlines():
        line = line.strip()
        inst, val_str = line.split()
        val = int(val_str)
        res.append((inst, val))

    return res


class Executor:
    """Emulator thingy for executing the instructions in the input"""

    def __init__(self, code, startat=0, acc=0) -> None:
        self.code = code
        self.i = startat
        self.acc = acc
        self.lines_run: set[int] = set([])
        self.keeprunning = True
        self.reached_end = False

    def iterate(self) -> None:
        if self.i in self.lines_run:
            self.keeprunning = False
            return
        else:
            self.lines_run.add(self.i)

        inst, val = self.code[self.i]

        if inst == "jmp":
            self.i += val
        else:
            if inst == "acc":
                self.acc += val
            self.i += 1
        self.reached_end = self.i == len(self.code)


    def run_all(self) -> int:
        while self.keeprunning:
            self.iterate()
        return self.acc
    #


def fix_and_run(instructions: list[tuple[str, int]]) -> int:
    """Identify the corrupt instruction, fix, and run"""
    G = nx.DiGraph()
    for u, (inst, val) in enumerate(instructions):
        acc = val if inst == 'acc' else 0
        G.add_node(u, acc=acc, inst=inst, val=val)

        increment = val if inst == "jmp" else 1
        v = u + increment
        G.add_edge(u, v)

    # Get the instructions that will eventually lead to the code terminating
    endnode = len(instructions)
    terminating_nodes = {n for n in G.nodes() if nx.has_path(G, n, endnode)}

    # Search for a node that can be modified to point to a terminating node
    fixed = False
    current_node = 0
    while not fixed:
        inst = nx.get_node_attributes(G, "inst")[current_node]
        val = nx.get_node_attributes(G, "val")[current_node]
        next_node = list(nx.neighbors(G, current_node))[0]

        # Find out where node could point to if we exchanged nop <-> jmp instr
        alternative = -1
        if inst == "nop":
            alternative = current_node + val
        elif inst == "jmp":
            alternative = current_node + 1

        # Change intruction if that would terminate the code
        if alternative in terminating_nodes:
            G.remove_edge(current_node, next_node)
            G.add_edge(current_node, alternative)
            fixed = True
        else:
            current_node = next_node
        #

    # Sum the increments to acc along the path
    path = nx.shortest_path(G, 0, endnode)
    res = sum(nx.get_node_attributes(G, 'acc')[node] for node in path[:-1])
    return res


def solve(data: str) -> tuple[int|str, ...]:
    instructions = parse(data)

    star1 = Executor(instructions).run_all()
    print(f"Solution to part 1: {star1}")

    star2 = fix_and_run(instructions)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 8
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
