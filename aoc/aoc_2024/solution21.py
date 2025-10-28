# .•·.*·`·  .   ·*+`.· •   ·+·.*`· .     + .`·*· `*·      ` · +. *.· `  ·`*·  .·
# · ` ` ·.•· ·  *.   `·+  *·   · Keypad Conundrum    .  * · +* ` .· ·  •`·· ·`+.
# `· *. + · ·* ·    ·+ https://adventofcode.com/2024/day/21    ·· *  +`·  ·.• `·
# ·`.· *· · *`·.   ·+   ··`*.`   .··*+  .  ·* .`·   ·•. ·· `.·  `·   *·.+·  `+·`


from functools import cache
import networkx as nx


def parse(s: str):
    return s.splitlines()


dirs = {
    "v": (1, 0),
    ">": (0, 1),
    "^": (-1, 0),
    "<": (0, -1),
}

dirs_inv = {v: k for k, v in dirs.items()}

numkeys = (
    ("7", "8", "9"),
    ("4", "5", "6"),
    ("1", "2", "3"),
    ("", "0", "A")
)

dirkeys = (
    ("", "^", "A"),
    ("<", "v", ">")
)


STARTAT = "A"


@cache
def _keys_as_graph(keys: tuple):
    """Makes a graph representing a keypad, where neighboring (i, j)-coords are connected."""

    # Add a node for each coordinate on the keypad with a valid symbol
    G = nx.Graph()
    for i, row in enumerate(keys):
        for j, char in enumerate(row):
            u = i, j
            if char == "":
                continue
            # Add position and the character
            G.add_node(u, char=char)
        #
    
    # Connect neighbors
    for u in G.nodes():
        i, j = u
        for di, dj in dirs.values():
            v = (i+di, j+dj)
            if v not in G.nodes():
                continue
            
            G.add_edge(u, v)
        #
    
    return G


def _represent_path_as_symbols(path: list) -> str:
    """Takes a path like [(2, 3), (2, 4), (1, 4), ...] and converts it to a string of symbols
    representing the steps taken along the path, e.g. '>^'."""

    res = ""
    for i in range(len(path)-1):
        a, b = path[i], path[i+1]
        delta = tuple(xb - xa for xa, xb in zip(a, b))
        symbol = dirs_inv[delta]
        res += symbol
    
    return res


@cache
def get_paths(a: str, b: str, directional: bool):
    """Returns a list of all shortest paths (represented as strings of direction symbols) from a to b
    on the specified keypad type."""
    
    # Make a graph over the keys
    keys = dirkeys if directional else numkeys
    G = _keys_as_graph(keys=keys)
    
    # Determine source and target nodes from the node attributes
    source = next(coord for coord, d in G.nodes(data=True) if d["char"] == a)
    target = next(coord for coord, d in G.nodes(data=True) if d["char"] == b)
    
    # Determine paths and convert to strings (and add 'A' to the end of each, as the key must be pressed)
    paths = list(nx.all_shortest_paths(G=G, source=source, target=target))
    res = [_represent_path_as_symbols(path)+STARTAT for path in paths]
    
    return res


@cache
def compute_len(a: str, b: str, depth=0, maxdepth: int|None=None) -> int:
    """Computes the shortest length of the sequence required to go from key a to key b, at a given
    'depth' of keypads. It is assumed that the 'deepest' keypad is the only numerical one."""
    
    # Set max depth to the one the method is initially called with
    if maxdepth is None:
        maxdepth = depth
    
    # Use directional numpad, except at the deepest 'layer'
    directional = depth != maxdepth
    
    # If we've reached the human-operated keypad, just return the shortest sequence.
    if depth == 0:
        paths = get_paths(a=a, b=b, directional=directional)
        return min(map(len, paths))
    
    # Otherwise, consider all possible paths and their lengths on the subsequent keypad
    paths = get_paths(a=a, b=b, directional=directional)
    best = -1
    
    for path in paths:
        # Add 'A' to the beginning of the sequence, as all robots start there
        sequence = STARTAT+path
        cost = 0
        
        # Recurse one level down for each sub-sequence
        for i in range(len(sequence) - 1):
            a2, b2 = sequence[i], sequence[i+1]
            cost += compute_len(a=a2, b=b2, depth=depth-1, maxdepth=maxdepth)
        
        if cost < best or best == -1:
            best = cost
    
    return best


def compute_code_length(code: str, depth: int):
    """Computes the optimal code length for the specified code, with n intermediate layers
    of robot-operated directional keypads."""
    
    # Add 'A' to beginning, as all robots start there
    sequence = STARTAT + code
    n = 0
    
    # Add the shortest length for each subsequence (all robots press 'A' at subsequence end, so they're independent)
    for i in range(len(sequence) - 1):
        a, b = sequence[i], sequence[i + 1]
        n_seq = compute_len(a, b, depth=depth)
        n += n_seq
    
    return n


def compute_complexity_sum(codes: list, depth=2):
    """Computes the sum over 'complexities' (numeric code part times shortest sequence length)
    over each code in the input, for the specified number of robot-operated intermediate keypads."""
    res = 0
    
    for code in codes:
        shortest = compute_code_length(code=code, depth=depth)
        numpart = int(''.join([c for c in code if c in "1234567890"]))
        res += (shortest*numpart)
        
    return res


def solve(data: str) -> tuple[int|str, int|str]:
    codes = parse(data)
    
    star1 = compute_complexity_sum(codes=codes, depth=2)
    print(f"Solution to part 1: {star1}")

    star2 = compute_complexity_sum(codes=codes, depth=25)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 21
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
