# ` · ··. `• ·.*` ··*    `· ·*. ·  .·`.   ·+ · *.•`  `·  ·+.*   ·`  ·.+ ·  `·.+·
# ·.*·  `  ·   .·*   .`·*· •`. · *·  . Duet ·  ` ·*  ·   `.· ·*     .`·    · ·*.
#  .·* •·`   `  ·. * · https://adventofcode.com/2017/day/18 ` .   *·  .·*`  +`··
# .·. ·` .*`·  ·* .  `· ·* . `* ·  ·. ` ·*  ` ··  .*` .   ` *·   ·   · `.•·.`*` 


from collections import defaultdict, deque


def parse(s: str):
    res = []
    for line in s.splitlines():
        parts = line.strip().split()
        for i, part in enumerate(parts):
            # Just convert to ints if possible. If that fails, keep as string
            try:
                parts[i] = int(parts[i])
            except ValueError:
                pass
            #
        res.append(tuple(parts))
    return res


class Register(defaultdict):
    """Helper class for registry data, so attempting to look up an int will just return the int"""
    def __missing__(self, key):
        return self.default_factory(key)


def _initialize_register(instructions: list, value: int = 0) -> Register:
    """Creates a register, using instructions to determine the appropriate registers and keys."""

    # Assume we just need a register for each string argument in the instructions
    keys = sorted(set([elem for elem in sum([list(tup[1:]) for tup in instructions], []) if isinstance(elem, str)]))

    reg = Register(lambda x: x)
    for k in keys:
        reg[k] = value

    return reg


def _update_reg(reg: Register, instruction: tuple) -> int:
    """Runs an instruction which modifies the registry."""

    # Parse args
    op, *args = instruction
    if len(args) == 1:
        x = args[0]
    elif len(args) == 2:
        x, y = args

    # Default to skipping one instuction ahead after running
    inc = 1

    if op == "set":
        reg[x] = reg[y]
    elif op == "add":
        reg[x] += reg[y]
    elif op == "mul":
        reg[x] *= reg[y]
    elif op == "mod":
        reg[x] = reg[x] % reg[y]
    elif op == "jgz":
        # Conditional jump
        if reg[x] > 0:
            inc = reg[y]
    else:
        raise ValueError

    return inc


def run_instructions(instructions: list):
    """Runs the instructions"""
    reg = _initialize_register(instructions)

    last_played = None

    ind = 0
    while 0 <= ind < len(instructions):
        inc = 1
        op, x, *_ = instructions[ind]

        if op == "snd":
            last_played = reg[x]  # Play sound with the frequency X
        elif op == "rcv":
            # If register x has non-zero value, return the last sound played
            if reg[x] != 0:
                return last_played
            #
        else:
            inc = _update_reg(reg, instructions[ind])

        ind += inc


def run_threads(instructions: list):
    """Threaded execution of the instructions, where snd and rcv sends and receives data to the other process."""

    # Define PIDs and the PID of the other process (PID1 for process 0 and vice versa)
    pids = [0, 1]
    others = [(i + 1) % len(pids) for i in range(len(pids))]
    pid = pids[0]  # Start with PID 0

    # Initiate registry
    regs = [_initialize_register(instructions) for _ in pids]
    for i, pid in enumerate(pids):
        regs[i]["p"] = pid

    # Define queues index of current instruction, and other stuff for each PID
    queues = [deque() for _ in pids]
    inds = [0 for _ in pids]
    n_sent = [0 for _ in pids]
    blocked = [False for _ in pids]

    deadlock = False

    while not deadlock:
        inc = 1
        ind = inds[pid]
        op, x, *_ = instructions[ind]

        if op == "snd":
            # Send data in register x to the pother process via its queue
            pid_other = others[pid]
            val = regs[pid][x]
            queues[pid_other].append(val)
            # Increment counter of how many pieces of data has been sent by each process
            n_sent[pid] += 1
        elif op == "rcv":
            try:
                # Attempt to receive data
                regs[pid][x] = queues[pid].popleft()
                blocked[pid] = False  # if successful, the process is unblocked
            except IndexError:
                # If unsuccessful, this process is blocked. Switch to the other one
                inc = 0
                blocked[pid] = True
                pid = others[pid]

                # If both processes are waiting for data from the other, it's a deadlock
                no_data = all(len(q) == 0 for q in queues)
                deadlock = no_data and all(blocked)
            #
        else:
            # If instruction is not send/receive, reuse the logic from part 1
            inc = _update_reg(regs[pid], instructions[ind])

        inds[pid] += inc

    res = n_sent[1]

    return res


def solve(data: str) -> tuple[int|str, int|str]:
    instructions = parse(data)

    star1 = run_instructions(instructions)
    print(f"Solution to part 1: {star1}")

    star2 = run_threads(instructions)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2017, 18
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()