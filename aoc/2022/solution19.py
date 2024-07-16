from copy import deepcopy
import re


def read_input():
    with open("input19.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = {}
    for line in s.split("\n"):
        d = {}
        blueprint_m = re.match(f'Blueprint (\d+)(:?.*)', line)
        blueprint = int(blueprint_m.group(1))

        prices_m = re.findall(r'Each (.*?) robot costs (.*?)\.', line)
        for material, price_string in prices_m:
            price = {}
            for snippet in price_string.split(" and "):
                n_part, mat_part = snippet.split(" ")
                n = int(n_part)
                price[mat_part] = n

            d[material] = price

        res[blueprint] = d

    return res


def get_initial_state(n_rounds=24):
    """Returns a starting state with just a single ore robot"""
    materials = ["ore", "clay", "obsidian", "geode"]
    state = dict(
        robots={k: 0 for k in materials},
        ressources={k: 0 for k in materials},
        rounds_left=n_rounds
    )
    state["robots"]["ore"] = 1

    return state


def represent_state(state):
    """Returns a state representation which is hashable"""
    keys = sorted(state.keys())
    res = tuple(tuple(tuple(state[k][kk] for kk in sorted(state[k])))
                if isinstance(state[k], dict) else state[k] for k in keys)
    return res


def buy_robot(state, blueprint, robot):
    """Pays for a new robot and starts building it"""
    state = deepcopy(state)
    price = blueprint[robot]
    for mat, n in price.items():
        state["ressources"][mat] -= n
        if state["ressources"][mat] < 0:
            raise ValueError("Can't afford to buy this.")
        #
    state["building"] = robot

    return state


def finish_building_inplace(state):
    """Finishes any robots being built"""
    try:
        mat = state["building"]
        state["robots"][mat] += 1
        del state["building"]
    except KeyError:
        pass


def collect_minerals_inplace(state):
    """Collects the minerals produced by the robots."""
    for type_, n in state["robots"].items():
        state["ressources"][type_] += n

    return state


def finish_turn_inplace(state):
    """Wraps up a turn by decrementing the rounds_left counter. Returns bool indicating whether 0 is reached."""
    state["rounds_left"] -= 1
    done = state["rounds_left"] <= 0
    return done


def get_possible_new_states(state, blueprint):
    """Generates a state for each robot which can be built from the current state, including not building any."""

    res = [deepcopy(state)]
    for robot in blueprint.keys():
        try:
            new_state = buy_robot(state, blueprint, robot)
            res.append(new_state)
        except ValueError:
            pass

    return res


def get_max_prices(blueprint):
    """Takes a blueprint and returns a dict mapping each material to the highest amount of that material it is possible
    to spend in a single turn."""
    res = {}
    for d in blueprint.values():
        for mat, price in d.items():
            res[mat] = max(res.get(mat, float("-inf")), price)
        #
    return res


def upper_bound(state, blueprint, objective="geode"):
    """Sets an upper bound on the number of objective material (geodes) that can be achieved from given state.
    This is done by letting a robot of every type (except the objective one) magically appear at each round, and
    buying an objective robot whenever possible."""

    non_objective = [k for k in state["robots"] if k != objective]
    state = deepcopy(state)
    while state["rounds_left"] > 0:
        # Try buying a robot producing the material we're after
        try:
            state = buy_robot(state, blueprint, objective)
        except ValueError:
            pass

        collect_minerals_inplace(state)
        # Magically increase all other robots by 1
        for mat in non_objective:
            state["robots"][mat] += 1

        finish_building_inplace(state)
        finish_turn_inplace(state)

    n = state["ressources"][objective]
    return n


def state_is_bad(state, blueprint, record):
    """Returns a bool indicating whether a state is 'bad'.
    States are bad if they have too many robots, i.e. robots that produce more (non geode) ressources than can be spent
    in a turn (this means we've wasted time and/or ressources).
    States are also 'bad' if they cannot possibly beat an existing record number of geodes."""

    max_prices = get_max_prices(blueprint)
    if any(state["robots"][mat] > maxprice for mat, maxprice in max_prices.items()):
        return True

    if upper_bound(state, blueprint) <= record:
        return True

    return False


def crunch(blueprints, blueprint_number, n_iterations=24):
    """Determines the max number of geodes which can be obtained with the given blueprint."""

    blueprint = blueprints[blueprint_number]
    state = get_initial_state(n_iterations)
    seen_states = set([])  # Maintain set of seen states so we can drop duplicates
    record = 0  # record number of geodes obtained so far
    cut = 1000  # cut between 'promising' states to grow and the remaining states

    # Define the 'growths' set from which we grow the number of states
    iterate_from = [state]
    i = 0  # iteration number, for printing progress
    msg_len_max = 0  # Max message length, for making printing look nice

    while iterate_from:
        i += 1
        new_states = []
        n_pruned = 0
        # Partition into promising (a) and non-promising (b) states
        a = iterate_from[:cut]
        b = iterate_from[cut:]

        for old_state in a:
            # Iterate over every state we can transition to from each promising state (every robot we can buy)
            for neighbor in get_possible_new_states(old_state, blueprint):
                # Drop state if we've already encountered it
                rep_ = represent_state(neighbor)
                if rep_ in seen_states:
                    continue

                # Finalize the turn (increase ressources, build robots, decrement rounds left)
                collect_minerals_inplace(neighbor)
                finish_building_inplace(neighbor)
                done = finish_turn_inplace(neighbor)

                # Prune the state (don't continue to grow it) if bad
                if state_is_bad(neighbor, blueprint=blueprint, record=record):
                    seen_states.add(rep_)
                    n_pruned += 1
                elif done:
                    # If the state has no rounds left, increase record if applicable, but do not grow the state
                    record = max(record, neighbor["ressources"]["geode"])
                else:
                    new_states.append(neighbor)
                    seen_states.add(rep_)

        # Collect the new states and the onew not grown. Order by available minerals (more and fancier is better)
        iterate_from = new_states + b
        order = ["geode", "obsidian", "clay", "ore"]
        iterate_from.sort(key=lambda d: tuple(d["ressources"][k] for k in order), reverse=True)
        try:
            record = max(record, iterate_from[0]["ressources"]["geode"])
        except IndexError:
            pass

        # Print status
        msg = f"Blueprint {blueprint_number}: "
        msg += f"{i} iterations, got {len(iterate_from)} states, pruned {n_pruned} and found {record} geodes."
        msg_len_max = max(msg_len_max, len(msg))
        msg += " "*(msg_len_max - len(msg))
        print(msg, end="\r")

    # Clear console and return record number of geodes
    print(msg_len_max*" ", end="\r")
    return record


def sum_quality_levels(blueprints):
    """Sums the 'quality level' (blueprint number * max_geodes) for all blueprints."""
    res = 0
    for k in sorted(blueprints.keys()):
        n_geodes = crunch(blueprints, k)
        quality_level = k*n_geodes
        res += quality_level

    print()
    return res


def product_max_geodes(blueprints, first_n=3, n_iterations=32):
    """Multiplies the max number of geodes for the initial n blueprints."""
    res = 1
    keys = sorted(blueprints.keys())[:first_n]
    for k in keys:
        n_geodes = crunch(blueprints, blueprint_number=k, n_iterations=n_iterations)
        res *= n_geodes

    return res


def main():
    raw = read_input()
    blueprints = parse(raw)

    star1 = sum_quality_levels(blueprints)
    print(f"Quality levels sum to {star1}.")

    star2 = product_max_geodes(blueprints)
    print(f"Product of max geodes in first three blueprints: {star2}.")


if __name__ == '__main__':
    main()
