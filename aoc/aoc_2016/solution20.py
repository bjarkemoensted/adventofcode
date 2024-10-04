def read_input():
    with open("input20.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = []
    for line in s.split("\n"):
        a, b = map(int, line.strip().split("-"))
        res.append((a, b))

    res.sort()
    return res


def find_allowed_IP_ranges(blacklist, lower=0, upper=4294967295, verbose=False):
    """Takes a blacklist (list of tuples) of ranges of forbidden IP adresses.
    Given the provided lower and upper bounds on valid IP adresses, returns a list of allowed ranges of IP addresses."""

    # Helper method for printing debugging stuff
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Start by assuming the entire range i allowed. Then iteratively remove the 'forbidden' intervals
    allowed = [(lower, upper)]

    for a, b in blacklist:
        next_allowed = []
        # Check if the current forbidden range overlaps with allowed region
        for x, y in allowed:
            vprint(f"Comparing {a}-{b} with {x}-{y}.")
            parts = []
            if a > y or b < x:
                # If there's no overlap at all, this allowed range is safe and proceeds to the next iteration
                next_allowed.append((x, y))
                vprint("miss")
            else:
                if x < a:
                    # Keep any region to the left of the the forbidden range
                    pre_part = (x, a - 1)
                    vprint(f"New allowed snippet: {pre_part}")
                    parts.append(pre_part)
                if y > b:
                    # Same for the right
                    post_part = (b + 1, y)
                    vprint(f"New allowed snippet: {post_part}")
                    parts.append(post_part)
                #
            # Done - proceed to the next part of the blacklist
            next_allowed += parts
            vprint()

        allowed = next_allowed
        vprint(f"New allowed list: {allowed}")

    return allowed


def main():
    raw = read_input()
    blacklist = parse(raw)

    allowed_IP_ranges = find_allowed_IP_ranges(blacklist=blacklist)
    star1 = min(a for a, _ in allowed_IP_ranges)
    print(f"The lowest allowed IP is {star1}.")

    star2 = sum(b - a + 1 for a, b in allowed_IP_ranges)
    print(f"There are {star2} allowed IP addresses.")


if __name__ == '__main__':
    main()
