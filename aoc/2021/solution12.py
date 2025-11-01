import copy
from collections import Counter

import networkx as nx

with open("input12.txt") as f:
    edges = [line.strip().split("-") for line in f]

G = nx.Graph()
for u, v in edges:
    G.add_edge(u, v)


def traverse_caves(G, allow_two_small_cave_visits=False):
    """Breadth-first traverses all possible paths in the caverns on the input graph.
    Returns all such paths.
    If allow_two_small_cave_visits, we allow a single small cave to be visited twice.
    Otherwise, small caves are visited at most once.
    """

    # Holds the paths we're still iterating on
    growing_paths = [['start']]
    # Holds the paths that have reached the 'end' node
    done_paths = []

    while growing_paths:
        # Extend all existing paths by one step
        new_paths = []
        for path in growing_paths:
            # Examine all neighbors of the last node in the path
            last_node = path[-1]
            for node in G.neighbors(last_node):
                # Always room for a small cave if we haven't been there yet
                allow_small_cave = node not in path
                # If we've been, but one double visit is allowed, check if we've already 'used' the double visit
                if not allow_small_cave and allow_two_small_cave_visits:
                    # Check if path contains double-visits to small caves
                    no_multivisits_yet = all(v <= 1 for v in Counter(s for s in path if s.islower()).values())
                    # If not, and if new node isn't the start node, we do allow it after all.
                    allow_small_cave = node != "start" and no_multivisits_yet

                # The new cave fits in path if it is large, or we have room for a small cave
                node_fits_in_path = node.isupper() or allow_small_cave
                if node_fits_in_path:
                    new_path = copy.deepcopy(path) + [node]
                    new_paths.append(new_path)
                #
            #

        # Mark any new paths that reach 'end' as done. Otherwise, keep iterating on them.
        growing_paths = []
        for path in new_paths:
            path_is_done = path[-1] == 'end'
            if path_is_done:
                done_paths.append(path)
            else:
                growing_paths.append(path)
            #
        #
    return done_paths


paths = traverse_caves(G)
print(f"Solution to star1: {len(paths)}.")

paths_with_double_visits = traverse_caves(G, allow_two_small_cave_visits=True)
print(f"Solution to star2: {len(paths_with_double_visits)}.")