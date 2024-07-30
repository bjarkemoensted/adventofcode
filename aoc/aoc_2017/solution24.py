from collections import Counter
import heapq


def parse(s):
    res = []
    for line in s.splitlines():
        a, b = map(int, line.split("/"))
        if a > b:
            a, b = b, a
        piece = (a, b)
        res.append(piece)

    res = tuple(sorted(res))

    return res


class MaxHeap:
    """Maxheap for use as a priority queue when searching"""
    def __init__(self):
        self._items = []

    def push(self, item, priority):
        heapq.heappush(self._items, (-priority, item))

    def pop(self):
        _, item = heapq.heappop(self._items)
        return item

    def __bool__(self):
        return bool(self._items)

    def __len__(self):
        return len(self._items)


def compute_strength(pieces):
    """Computes the strength of the given pieces"""
    if isinstance(pieces[0], int):
        pieces = [pieces]
    res = sum(a+b for a, b in pieces)
    return res


def heuristic(state, longer_is_better=False):
    """Returns the strength of all remaining pieces (optimistic estimate of the max additional strength possible
    be adding remaining pieces)"""
    endpoint, pieces = state
    h = len(pieces) if longer_is_better else compute_strength(pieces)
    return h


def grow_single(state):
    """Iterates over the states it is possible to reach from the input state"""
    endpoint, pieces = state
    for i, piece in enumerate(pieces):
        if endpoint in piece:
            a, b = piece
            new_endpoint = b if endpoint == a else a

            new_remaining = tuple(otherpiece for ii, otherpiece in enumerate(pieces) if ii != i)
            new_state = (new_endpoint, new_remaining)
            yield piece, new_state
        #
    #


def build_bridge(pieces, longer_is_better=False, maxiter=None):
    """Builds the Best Bridge^tm using the provided pieces. Runs an A* kinda algorithm.
    If longer_is_better, the algorithm optimizes for length, rather than strength.
    Whether length or strength is chosen, the opposite metric is used as a tiebreaker."""

    if maxiter is None:
        maxiter = float("inf")

    # Set up stuff for A*
    initial_state = (0, pieces)
    priorities = {initial_state: 0}
    h0 = heuristic(initial_state, longer_is_better=longer_is_better)
    d_f = {initial_state: h0}

    # Stuff for reconstructing the bridge, given states
    camefrom = dict()
    last_added = dict()

    queue = MaxHeap()
    queue.push(initial_state, priority=h0)

    nits = 0
    while queue:
        current = queue.pop()

        for added_piece, new_state in grow_single(current):
            # If we care about length, adding a pieces increases by one. Otherwise, it depends on the piece strength
            d_uv = 1 if longer_is_better else compute_strength(added_piece)
            new_score = priorities[current] + d_uv

            # Update if we beat the record
            improved = new_score > priorities.get(new_state, float("-inf"))
            if improved:
                priorities[new_state] = new_score
                camefrom[new_state] = current
                last_added[new_state] = added_piece
                f = new_score + heuristic(initial_state, longer_is_better=longer_is_better)
                d_f[new_state] = f
                queue.push(new_state, f)
            #

        nits += 1
        if nits >= maxiter:
            break
        #

    def reconstruct(node):
        """Reconstructs the path (bridge) from a given end state"""
        link = node
        path = []
        while link in camefrom:
            path.append(last_added[link])
            link = camefrom[link]

        return path[::-1]

    # Determine the Best Bridge^tm, keeping ties if any occur
    tied = []
    record = float("-inf")
    for state, score in priorities.items():
        if score > record:
            record = score
            tied = [state]
            continue
        if score == record:
            tied.append(state)
        #

    # Whatever the metric for 'best' (strength or length), use the other metric to break ties
    tiebreaker = lambda state: compute_strength(reconstruct(state)) if longer_is_better else len(reconstruct(state))
    best = max(tied, key=tiebreaker)

    res = reconstruct(best)
    return res


def solve(data: str):
    pieces = parse(data)

    bridge = build_bridge(pieces)
    star1 = compute_strength(bridge)
    print(f"Solution to part 1: {star1}")

    long_bridge = build_bridge(pieces, longer_is_better=True)
    star2 = compute_strength(long_bridge)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 24
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)

    solve(raw)


if __name__ == '__main__':
    main()
