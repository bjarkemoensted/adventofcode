def read_input():
    with open("input19.txt") as f:
        puzzle_input = f.read()

    return puzzle_input


def parse(s):
    wfpart, partpart = s.split("\n\n")

    rules = {}
    parts = []

    # Parse the workflows
    for line in wfpart.split("\n"):
        cut = line.index("{")
        k = line[:cut]
        snippets = line[cut+1:-1].split(",")
        rules[k] = snippets

    # Parse the part info
    for line in partpart.split("\n"):
        d = {}
        for snippet in line[1:-1].split(","):
            k, v = snippet.split("=")
            d[k] = int(v)
        parts.append(d)

    return rules, parts


def apply_rule(part, rule):
    """Applies a rule to a given part, i.e. returns the destination for a part given a rule"""
    for line in rule[:-1]:
        cond, res = line.split(":")
        var = cond[0]

        fs = f"lambda d: d['{var}']{cond[1:]}"
        f = eval(fs)
        if f(part):
            return res
    return rule[-1]


def follow_workflow(part, rules):
    """Iteratively applies rules to a part, until its state is either accepted ('A') or rejected ('R')."""
    k = "in"
    while k not in ("A", "R"):
        k = apply_rule(part, rules[k])

    return k


def sum_accepted_ratings(parts, rules):
    res = 0
    for part in parts:
        final = follow_workflow(part, rules)
        if final == "A":
            res += sum(part.values())
        #

    return res


def count_elements_in_space(d):
    res = 1
    for a, b in d.values():
        res *= (b - a)
    return res


def apply_rule_to_sets(part, rule):
    """Takes a set of parts, specified as e.g. {'x': (1, 4001), ...}.
    Returns a dict where the resulting states are keys, and the values are the subsets moving to those states."""

    vol1 = count_elements_in_space(part)
    res = {}
    # Continually updated remaining subset of parts
    part = {k: v for k, v in part.items()}
    for rulestring in rule:
        # If rule is the last one, there's no condition. The remaining subset satisfies this
        last_rule = ":" not in rulestring
        if last_rule:
            res[rulestring] = res.get(rulestring, []) + [part]
            continue

        cond, dest = rulestring.split(":")
        var = cond[0]
        set_ = part[var]
        a, b = set_
        comp = cond[1]
        cut = int(cond[2:])

        # Determine the subset of the relevant variable which satisfies the condition
        if comp == ">":
            bad = (a, cut+1)
            good = (cut+1, b)
        elif comp == "<":
            good = (a, cut)
            bad = (cut, b)
        else:
            raise ValueError

        # The subset satisfying this condition move to the resulting state. The remainder tries the next rule
        good_parts = {k: good if k == var else v for k, v in part.items()}
        res[dest] = res.get(dest, []) + [good_parts]
        part[var] = bad

    # Make sure we didn't miss any subsets
    vol2 = sum(map(count_elements_in_space, sum(res.values(), [])))
    if vol1 != vol2:
        raise ValueError

    return res


def count_total_accepted(rules):
    """Takes the input state and total space of parts considered. Iteratively breaks down into subsets until all are
    accepted or rejected."""

    d = {k: (1, 4001) for k in "xmas"}
    states = {"in": [d]}
    while any(k not in ("A", "R") for k in states.keys()):
        newstates = {}
        for k, dicts in states.items():
            # Keep accepted/rejected subsets
            if k in ("A", "R"):
                newstates[k] = dicts
                continue
            # Apply rules to the remaining parts
            rule = rules[k]
            for d in dicts:
                dest2subsets = apply_rule_to_sets(part=d, rule=rule)
                for dest, subsets in dest2subsets.items():
                    newstates[dest] = newstates.get(dest, []) + subsets
                #
            #
        states = newstates

    res = sum(map(count_elements_in_space, states["A"]))
    return res


def main():
    raw = read_input()
    rules, parts = parse(raw)

    part = parts[0]
    follow_workflow(part, rules)

    star1 = sum_accepted_ratings(parts, rules)
    print(f"Rating sum is {star1}.")

    star2 = count_total_accepted(rules)
    print(f"Total number of approved parts: {star2}.")


if __name__ == '__main__':
    main()
