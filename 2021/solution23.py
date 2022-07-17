import numpy as np

example = \
"""#############
#...........#
###B#C#B#D###
  #A#D#C#A#
  #########"""


# The extra snippet that appears when the paper is unfolded
extra_snippet = \
"""  #D#C#B#A#
  #D#B#A#C#"""


with open("input23.txt") as f:
    data = f.read()


def parse(s, unfold=False):
    """Converts a puzzle input to a numpy nd array of chars"""
    lines = s.split("\n")

    if unfold:
        addition = extra_snippet.split("\n")
        lines = lines[:3] + addition + lines[3:]

    maxlength = max(len(line) for line in lines)
    map_ = np.full(shape=(len(lines), maxlength), fill_value=" ", dtype=str)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            map_[i, j] = char

    return map_


def clear_map(map_chars):
    n_rows, n_cols = map_chars.shape
    res = np.full(shape=(n_rows, n_cols), fill_value=" ", dtype=str)
    for i in range(n_rows):
        for j in range(n_cols):
            c = map_chars[i, j]
            res[i, j] = "." if c in "ABCD" else c
        #

    return res


def make_graph_from_map(map_chars):
    """Construct a graph from a map of an amphipod cave thingy.
    Returns a dict mapping the coordinates of each empty space to neighboring empty coordinates"""
    n_rows, n_cols = map_chars.shape
    space_chars = set("ABCD.")  # These characters indicate a spot with space for an amphipod
    coords_with_spaces = [(i, j) for i in range(n_rows) for j in range(n_cols) if map_chars[i, j] in space_chars]

    G = {}
    for coords in coords_with_spaces:
        i0, j0 = coords
        for delta_i in (-1, 0, 1):
            for delta_j in (-1, 0, 1):
                neighbor_coords = (i0 + delta_i, j0 + delta_j)
                # Don't connect nodes to themselves, and don't allow diagonal moves
                if abs(delta_i) + abs(delta_j) in (0, 2):
                    continue
                if neighbor_coords in coords_with_spaces:
                    try:
                        G[coords].add(neighbor_coords)
                    except KeyError:
                        G[coords] = {neighbor_coords}
                    #
                #
            #
        #
    return G


def make_state_hashable(state):
    """A state is a tuple of
    1) A positions dict denoting the positions of all amphipods like {"A1": (1,1), ...}
    2) A set of the amphipods that have arrived home and cannot move again, e.g. {'A1', 'B2', ...}.
    This converts the state into a hashable format for use as graph nodes."""

    positions, gone_home = state
    positions_tuple = tuple(sorted(positions.items()))
    gone_home_tuple = tuple(sorted(gone_home))

    hashable_state = (positions_tuple, gone_home_tuple)
    return hashable_state


def recover_state(hashable_state):
    positions_tuple, gone_home_tuple = hashable_state
    positions = dict(positions_tuple)
    gone_home_set = {amph for amph in gone_home_tuple}

    state = (positions, gone_home_set)
    return state


def determine_amphipod_positions(map_chars):
    """Determines the locations of amphipods from a map of their burrow.
    Returns e.g. {"B1": (1,1), ...}, indicating a bronze amphipod at location (1,1)."""

    locations = {}
    amphipod_letters = set("ABCD")
    suffices = {letter: 1 for letter in amphipod_letters}
    n_rows, n_cols = map_chars.shape
    for i in range(n_rows):
        for j in range(n_cols):
            char = map_chars[i, j]
            if char in amphipod_letters:
                locations[char+str(suffices[char])] = (i, j)
                suffices[char] += 1
            #
        #

    return locations


def breadth_first(graph, starting_node, forbidden_nodes=None, nostop_nodes=None):
    """Does breadth-first iteration on input graph (dict structure). Terminates paths at forbidden nodes.
    Does not allow paths to stop at nostop nodes, but allows paths to pass through.
    Returns a dict like {endnode: cost}"""

    if forbidden_nodes is None:
        forbidden_nodes = set([])
    if nostop_nodes is None:
        nostop_nodes = {(1, 3), (1, 5), (1, 7), (1, 9)}

    paths_done = []
    paths_running = [[starting_node]]
    shortest_distances = {}  # Maps path end nodes to the shortest distance to them

    def allowed_next_steps(path_part):
        nonlocal forbidden_nodes
        neighbors = graph.get(path_part[-1], set([]))
        return neighbors - forbidden_nodes - set(path_part)

    while paths_running:
        new_paths = []
        # Try extending all the paths we're considering with their neighbors
        for path in paths_running:
            next_steps = allowed_next_steps(path)
            if not next_steps:
                # If we can't extend the path, we're done with it
                paths_done.append(path)
                continue
            # If we can extend the path, make all the possible extensions
            for next_ in next_steps:
                new_path = [node for node in path] + [next_]
                new_paths.append(new_path)
                # If we're allowed to stop at this node, update the shortest distances dict
                if next_ not in nostop_nodes:
                    dist = len(new_path) - 1
                    shortest_distances[next_] = min(dist, shortest_distances.get(next_, float('inf')))

            #
        paths_running = new_paths

    return shortest_distances


def display_state(state, map_chars):
    positions, gone_home_set = state
    if isinstance(positions, tuple):
        positions = dict(positions)

    map_with_amphipods = map_chars.copy()
    for amph, (i, j) in positions.items():
        map_with_amphipods[i, j] = amph[0]
    lines = ["".join(line) for line in map_with_amphipods]
    lines.append(f"Gone home: {', '.join(sorted(gone_home_set)) if gone_home_set else 'None'}.")
    print("\n".join(lines))


chars = parse(data)
map_ = clear_map(chars)
amphipod_positions = determine_amphipod_positions(chars)

burrow = make_graph_from_map(map_)

# Store the nodes, hallway nodes, and roome nodes
all_nodes = set(burrow.keys())
hallway_nodes = {node for node in all_nodes if node[0] == 1}
room_nodes = all_nodes - hallway_nodes

# Map each amphipod type to the set of its home nodes
letter2column = dict(zip("ABCD", [3, 5, 7, 9]))
letter2home = {letter: {n for n in room_nodes if n[1] == letter2column[letter]} for letter in "ABCD"}


def determine_forbidden_and_nostop_nodes(amphipod, positions):
    """amphipod is like 'A1', positions is like {"A1": (1, 1), ...}"""
    assert len(amphipod) == 2
    amphipod_type = amphipod[0]
    pos = positions[amphipod]

    # Positions where other amphipods are, are forbidden no matter what
    forbidden = {v for k, v in positions.items() if k != amphipod}

    home = letter2home[amphipod_type]
    foreign_rooms = room_nodes - home

    # Room where the amphipod currently resides
    current_room = [set_ for set_ in letter2home.values() if pos in set_]
    if len(current_room) > 1:
        raise ValueError
    current_room = current_room[0] if current_room else set([])

    nostop_nodes = {(1, 3), (1, 5), (1, 7), (1, 9)}
    nostop_nodes.update(foreign_rooms)
    nostop_nodes.update(current_room)

    if pos in home:
        forbidden.update(foreign_rooms)
        nostop_nodes.update(home)
    elif pos in room_nodes:
        # If amphipod is in a room, it is allowed to go to the hallway, or go to its home
        forbidden.update(foreign_rooms - current_room)
    elif pos in hallway_nodes:
        # if in the hallway, the amphipod is forbidden to go anywhere away from home
        nostop_nodes.update(hallway_nodes)
        forbidden.update(foreign_rooms)
    else:
        raise ValueError

    return forbidden, nostop_nodes


def get_movement_cost(amphipod):
    d = dict(zip("ABCD", [1, 10, 100, 1000]))
    letter = amphipod[0]
    return d[letter]


def is_home(amphipod, amphipod_pos):
    letter = amphipod[0]
    return amphipod_pos in letter2home[letter]


def state_is_final(state):
    positions, _ = state
    return all(is_home(amph, pos) for amph, pos in positions.items())


def get_possible_new_states(state):
    """Gives (new state, cost to get there). Returns None if no new states are possible."""
    positions, gone_home = state
    res = []

    for amphipod, coords in positions.items():
        # Amphipods can't move after reaching their home
        if amphipod in gone_home:
            continue

        forbidden, nostop = determine_forbidden_and_nostop_nodes(amphipod=amphipod, positions=positions)
        new_edges = breadth_first(graph=burrow, starting_node=coords, forbidden_nodes=forbidden, nostop_nodes=nostop)
        for new_coords, distance in new_edges.items():
            cost = distance*get_movement_cost(amphipod)
            new_positions = {k: v for k, v in positions.items()}
            new_positions[amphipod] = new_coords
            new_gone_home = {amph for amph in gone_home}
            if is_home(amphipod, new_coords):
                new_gone_home.add(amphipod)

            new_state = (new_positions, new_gone_home)
            res.append((new_state, cost))
        #

    return res


def state_is_doomed(state):
    """Returns true if a state is doomed to fail, i.e. it is not possible to reach the desired end state."""
    positions, gone_home = state

    coords2type_ = {coords: amph[0] for amph, coords in positions.items()}
    for amphipod in gone_home:
        i0, j0 = positions[amphipod]
        lowest_point = map_.shape[0] - 1
        # The 'danger zone' below this amphipod is blocked by it and so has to be filled by the same type.
        danger_zone = [(i, j0) for i in range(i0 + 1, lowest_point)]
        type_ = amphipod[0]
        if any(coords2type_.get(coords, '.') != type_ for coords in danger_zone):
            return True

    return False


def build_paths(initial_state):
    paths = [([make_state_hashable(initial_state)], [0])]
    done = False
    found_solution = False
    cheapest_path = float('inf')
    n_iterations = 0
    hashed_state2cheapest_cost = {make_state_hashable(initial_state): 0}
    longest_message_length = 0  # For printing statuses

    while not done:
        n_iterations += 1
        # If we've already found a solution, keep growing the cheaper paths uwe have the optimal one
        if found_solution:
            paths_grow = [(path, costs) for path, costs in paths if costs[-1] < cheapest_path]
            paths_frozen = [(path, costs) for path, costs in paths if costs[-1] >= cheapest_path]
            done = not paths_grow
        # Otherwise just grow some of the cheaper ones until we have a solution
        else:
            paths_grow = paths
            paths_frozen = []

        n_paths_grow = len(paths_grow)
        new_paths = []
        n_skipped_suboptimal = 0
        for path in paths_grow:
            # Extend each of the paths we're currently growing
            steps, costs = path
            state = recover_state(steps[-1])

            # Examine all new possible states found by extending the last step
            for new_state, extra_cost in get_possible_new_states(state):
                if state_is_doomed(new_state):
                    continue
                new_total = costs[-1] + extra_cost
                key = make_state_hashable(new_state)

                # Don't use an extension if it leads to a state which is reacable in a cheaper way
                if new_total < hashed_state2cheapest_cost.get(key, float('inf')):
                    hashed_state2cheapest_cost[key] = new_total

                    # Get the updated path infp
                    new_steps = steps + [key]
                    new_costs = costs + [new_total]
                    new_path = (new_steps, new_costs)
                    new_paths.append(new_path)

                    # Check if this path solves the problem
                    if state_is_final(new_state):
                        found_solution = True
                        cheapest_path = min(cheapest_path, new_total)
                    #
                else:
                    n_skipped_suboptimal += 1
            #

        # Update the set of paths being considered
        paths = new_paths + paths_frozen
        # If any path visits a state using a suboptimal route, drop it again (a cheaper alternative must exist)
        for i in range(len(paths) - 1, -1, -1):
            steps, costs = paths[i]
            if any(cost > hashed_state2cheapest_cost[step] for step, cost in zip(steps, costs)):
                del paths[i]
                n_skipped_suboptimal += 1
        paths.sort(key=lambda tup: tup[1])

        msg = f"Did {n_iterations} iterations. Considering {len(paths)} paths. Grew {n_paths_grow} paths."
        msg += f" Distance: {min(path[1][-1] for path in paths)}/{cheapest_path}."

        # Make sure the message clears the preceding message from screen
        longest_message_length = max(longest_message_length, len(msg))
        msg += (longest_message_length - len(msg))*' '
        print(msg, end="\r")
    print()

    solutions = []
    for path in paths:
        steps, costs = path
        state = recover_state(steps[-1])
        if state_is_final(state):
            solutions.append(path)

    return solutions


def get_energy_minimum(paths):
    lowest_cost = min(path[1][-1] for path in paths)
    return lowest_cost


initial_state = (amphipod_positions, set([]))

solutions = build_paths(initial_state)
star1 = get_energy_minimum(solutions)
print(f"Most efficient solution costs {star1} energy.")


### Star 2
print()
chars = parse(data, unfold=True)
map_ = clear_map(chars)
amphipod_positions = determine_amphipod_positions(chars)

burrow = make_graph_from_map(map_)

# Store the nodes, hallway nodes, and roome nodes
all_nodes = set(burrow.keys())
hallway_nodes = {node for node in all_nodes if node[0] == 1}
room_nodes = all_nodes - hallway_nodes

# Map each amphipod type to the set of its home nodes
letter2column = dict(zip("ABCD", [3, 5, 7, 9]))
letter2home = {letter: {n for n in room_nodes if n[1] == letter2column[letter]} for letter in "ABCD"}

initial_state = (amphipod_positions, set([]))

solutions = build_paths(initial_state)
star2 = get_energy_minimum(solutions)
print(f"Most efficient solution for problem 2 costs {star2} energy.")
