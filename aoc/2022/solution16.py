from copy import deepcopy
import re


def read_input():
    with open("input16.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


test = """Valve AA has flow rate=0; tunnels lead to valves DD, II, BB
Valve BB has flow rate=13; tunnels lead to valves CC, AA
Valve CC has flow rate=2; tunnels lead to valves DD, BB
Valve DD has flow rate=20; tunnels lead to valves CC, AA, EE
Valve EE has flow rate=3; tunnels lead to valves FF, DD
Valve FF has flow rate=0; tunnels lead to valves EE, GG
Valve GG has flow rate=0; tunnels lead to valves FF, HH
Valve HH has flow rate=22; tunnel leads to valve GG
Valve II has flow rate=0; tunnels lead to valves AA, JJ
Valve JJ has flow rate=21; tunnel leads to valve II"""


def parse(s):
    d = {}
    for line in s.split("\n"):
        match = re.match(r'Valve (.*) has flow rate=(.*); (.*)', line)
        hits = match.groups()
        node = {}
        name = hits[0]
        node["flow"] = int(hits[1])

        node["neighbors"] = sorted(match.groups()[-1].replace(",", "").split(" ")[4:])

        d[name] = node
    return d


def get_initial_state(positions=None, minutes_left=30):
    if positions is None:
        positions = ["AA"]
    d = dict(
        released=0,
        valves_open=set([]),
        positions=positions,
        minutes_left=minutes_left
    )

    return d


def let_off_some_steam(state, G):
    for valve in state["valves_open"]:
        flow = G[valve]["flow"]
        state["released"] += flow

    return


def tick(state):
    state["minutes_left"] -= 1
    return


def represent_state(state, include_time=True):
    """Gives a canonical, hashable representation of the input state"""

    keys = sorted(state.keys())
    if not include_time:
        time_key = "minutes_left"
        assert time_key in keys
        keys = [k for k in keys if k != time_key]

    def convert(elem): return elem if isinstance(elem, int) else tuple(sorted(elem))
    res = tuple(convert(state[k]) for k in keys)
    return res


def open_valve(state, valve):
    """Returns a copy of input states, but where the specified valve is open"""
    if valve in state["valves_open"]:
        raise ValueError(f"Valve {valve} is already open")
    res = deepcopy(state)
    res["valves_open"].add(valve)
    return res


def get_possible_new_states(old_state, G):
    grow = [deepcopy(old_state)]
    inds = range(len(old_state["positions"]))
    for ind in inds:
        grown_states = []
        for growstate in grow:
            pos = growstate["positions"][ind]

            # Try opening the valve at the current position
            if G[pos]["flow"] > 0:
                try:
                    opened = open_valve(growstate, pos)
                    grown_states.append(opened)
                except ValueError:
                    pass

            # Move to neighboring sites
            for neighbor in G[pos]["neighbors"]:
                new_state = deepcopy(growstate)
                new_state["positions"][ind] = neighbor
                grown_states.append(new_state)
            #
        grow = grown_states

    return grow


def state_is_bad(state, G, record):
    """Returns true if the input state will never be able to outperform the current record"""
    flow = sum(G[valve]["flow"] for valve in state["valves_open"])
    upper_bound = state["released"]
    closed = set(G.keys()) - state["valves_open"]
    remaining_flows = [G[valve]["flow"] for valve in closed]
    ordered = sorted(filter(lambda x: x > 0, remaining_flows))

    n_actors = len(state["positions"])
    for i in range(state["minutes_left"]):
        upper_bound += flow
        if (i % 2) == 0:
            for _ in range(n_actors):
                if ordered:
                    flow += ordered.pop()
                #
            #
        #

    return upper_bound <= record


def crunch(initial_state, G):
    """Determines the max amount of steam that can be released"""

    seen_states = set([])
    state2earliest_occurrence = {}

    record = 0
    cut = 1000  # cut between 'promising' states to grow and the remaining states

    # Define the 'growth' set from which we grow the number of states
    iterate_from = [deepcopy(initial_state)]
    i = 0  # iteration number, for printing progress

    msg_max_len = 0

    while iterate_from:
        i += 1
        new_states = []
        n_pruned = 0
        # Partition into promising (a) and non-promising (b) states
        a = iterate_from[:cut]
        b = iterate_from[cut:]

        for old_state_base in a:
            old_state = deepcopy(old_state_base)
            let_off_some_steam(old_state, G)
            for new_state in get_possible_new_states(old_state, G):
                tick(new_state)
                # Drop state if we've already encountered it
                rep_ = represent_state(new_state)
                if rep_ in seen_states:
                    continue

                timeless = represent_state(new_state, include_time=False)
                time_left = new_state["minutes_left"]
                if state2earliest_occurrence.get(timeless, float("inf")) <= time_left:
                    continue
                else:
                    state2earliest_occurrence[timeless] = time_left

                done = new_state["minutes_left"] <= 0
                # Prune the state (don't continue to grow it) if bad
                if state_is_bad(new_state, G, record=record):
                    seen_states.add(rep_)
                    n_pruned += 1
                elif done:
                    # If the state has no rounds left, increase record if applicable, but do not grow the state
                    record = max(record, new_state["released"])
                else:
                    new_states.append(new_state)
                    seen_states.add(rep_)
                #
            #

        # Collect the new states and the ones not grown. Order by total steam released
        iterate_from = new_states + b

        iterate_from.sort(key=lambda d: d["released"], reverse=True)

        try:
            record = max(record, iterate_from[0]["released"])
        except IndexError:
            pass

        msg = f"Round {i}, growing {len(iterate_from)} states. Pruned {n_pruned}. Current record: {record}."
        msg_max_len = max(msg_max_len, len(msg))
        msg += " "*(msg_max_len - len(msg))
        print(msg, end="\r")

    print(" "*msg_max_len, end="\r")

    return record


def main():
    raw = read_input()
    G = parse(raw)

    initial_state = get_initial_state(positions=["AA"], minutes_left=30)
    released = crunch(initial_state, G)
    print(f"The maximum amount of steam that can be released is {released}.")

    initial_state_w_elephant = get_initial_state(positions=["AA", "AA"], minutes_left=26)
    released_w_elephant = crunch(initial_state_w_elephant, G)
    print(f"The maximum amount of steam that can be released with an elephant helper is {released_w_elephant}.")


if __name__ == '__main__':
    main()
