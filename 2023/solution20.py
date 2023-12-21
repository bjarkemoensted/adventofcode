import abc
import math
import networkx as nx
from typing import final

test1 = """broadcaster -> a, b, c
%a -> b
%b -> c
%c -> inv
&inv -> a"""


def read_input():
    with open("input20.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = []
    for line in s.split("\n"):
        type_ = line[0]
        if type_ in ("%", "&"):
            source, dests_str = line[1:].split(" -> ")
            dests = tuple(dests_str.split(", "))
        elif line.startswith("broadcaster"):
            source = "broadcaster"
            type_ = "broadcaster"
            dests = tuple(line.split(" -> ")[-1].split(", "))
        else:
            raise ValueError(line)
        res.append((type_, source, dests))

    return res


class State(abc.ABC):
    prefix = ""

    @final
    def __init__(self, label: str, verbose=False, keep_history=True):
        self.label = label
        self.data = None
        self.verbose = verbose
        self.keep_history = keep_history
        self._history = []
        self.last_pulse = None

    @abc.abstractmethod
    def set_data(self, sources):
        raise NotImplementedError

    @abc.abstractmethod
    def ping(self, source, val):
        raise NotImplementedError

    @final
    def __hash__(self):
        return hash(self.label)

    @final
    def __call__(self, source, val):
        if self.keep_history:
            self._history.append(val)
        if self.data is None:
            raise ValueError
        if self.verbose:
            pulse = {0: "low", 1: "high"}[val]
            source_str = source.label if isinstance(source, State) else source
            print(f"{source_str} -{pulse}-> {self.label}")

        res = self.ping(source, val)
        self.last_pulse = res
        return res

    def __repr__(self):
        s = f"{self.prefix}{self.label}"
        return s

    def __str__(self):
        ds = "Contains no data" if self.data is None else f"Data: {self.data}"
        s = f"{repr(self)} ({self.__class__.__name__}). {ds}."
        return s


class Flipflop(State):
    prefix = "%"

    def set_data(self, sources):
        self.data = 0

    def ping(self, source, val):
        if val == 0:
            self.data = (self.data + 1) % 2
            return self.data


class Conjunction(State):
    prefix = "&"

    def set_data(self, sources):
        self.data = {k: 0 for k in sources}

    def ping(self, source, val):
        self.data[source] = val
        return 1 - min(self.data.values())


class Broadcaster(State):
    def set_data(self, sources):
        self.data = -1

    def ping(self, source, val):
        return val


class Output(State):
    def set_data(self, sources):
        self.data = -1

    def ping(self, source, val):
        if val == 0:
            self.data = 1
        return None


def make_state(type_, label, *args, **kwargs):
    cons = {"%": Flipflop, "&": Conjunction}
    con = cons.get(type_, Broadcaster)
    res = con(label=label, *args, **kwargs)
    return res


def build_network(data, *args, **kwargs):
    G = nx.DiGraph()

    for type_, source, dests in data:
        for dest in dests:
            G.add_edge(source, dest)
        state = make_state(type_, source, *args, **kwargs)
        G.nodes[source]["state"] = state

    for n in G.nodes():
        if "state" not in G.nodes[n]:
            G.nodes[n]["state"] = Output(n)

        G.nodes[n]["state"].set_data(G.predecessors(n))

    return G


def count_pulses(G):
    """Counts the number of low/high pulses each module has received"""
    low_highs = [0, 0]
    for node in G.nodes:
        for i in range(len(low_highs)):
            low_highs[i] += sum(v == i for v in G.nodes[node]["state"]._history)
        #

    return tuple(low_highs)


def push_button(G, n_times=1):
    """Pushes the button module a desired number of times. Returns a list of tuples (source, destination, value)
    of all pulses sent as a result of the button press."""

    all_pulses = []
    for _ in range(n_times):
        # Start with a pulse from the button to the broadcaster. The repeatedly handle new pulses
        pulses = [("button", "broadcaster", 0)]
        while pulses:
            all_pulses += pulses
            next_pulses = []
            for source, dest, val in pulses:
                newval = G.nodes[dest]["state"](source, val)
                if newval is not None:
                    for neighbor in G.successors(dest):
                        next_pulse = (dest, neighbor, newval)
                        next_pulses.append(next_pulse)

            pulses = next_pulses
        #
    return all_pulses


def compute_button_presses_until_low_pulse(G):
    """Computes the number of button presses needed until the output module receives a low pulse.
    The trick is to spot that the graph has 4 distinct submodules which operate independently.
    Each submodule repeats its state pattern after emitting a low pulse, so the solution is simply the
    least common multiple of the number of steps it takes each submodule to fire a low pulse."""

    hubs = ["vg", "vc", "nb", "ls"]  # found by inspecting the graph

    # Start by identifying cycle lengths
    cycles = [None for _ in hubs]
    n = 0
    while any(elem is None for elem in cycles):
        pulses = push_button(G, n_times=1)
        n += 1
        for i, cycle in enumerate(cycles):
            if cycle is not None:
                continue
            hub = hubs[i]
            if any(dest == hub and val == 0 for _, dest, val in pulses):
                cycles[i] = n
            #
        #

    res = math.lcm(*cycles)
    return res


def main():
    raw = read_input()
    data = parse(raw)

    G1 = build_network(data, verbose=False)
    push_button(G1, n_times=1000)
    counts = count_pulses(G1)
    star1 = counts[0]*counts[1]
    print(f"Product of number of low and high pulses is {star1}.")

    G2 = build_network(data)
    star2 = compute_button_presses_until_low_pulse(G2)
    print(f"The output module receives its first low pulse after {star2} button presses.")


if __name__ == '__main__':
    main()
