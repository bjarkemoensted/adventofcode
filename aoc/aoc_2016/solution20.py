# ꞏ⸳`•  ꞏ+⸳ꞏ    +  ꞏ  `⸳.*  ꞏ+ .  `  +  ꞏ  ⸳.ꞏ.  ` ⸳ꞏ+ ⸳    + ꞏꞏ  `    .  *ꞏ+⸳ `
#   ⸳. ⸳   *  ꞏ ..`* `    . ⸳`ꞏ•  Firewall Rules  ꞏ+.   + ꞏ ` .  * ⸳` ꞏ+   *. .ꞏ
#   *ꞏ.    ꞏ⸳•` ⸳    * https://adventofcode.com/2016/day/20 .   +.ꞏ` ⸳  ..   ꞏ  
# ⸳ꞏ.`*`.ꞏ+ .`+⸳ +  .ꞏ  ⸳•  `ꞏ*⸳  .  *` .ꞏ⸳ ꞏ*    . ` ⸳ꞏ  . ⸳ `ꞏ.⸳.*+` ꞏ`    + ⸳


def parse(s):
    res = []
    for line in s.split("\n"):
        a, b = map(int, line.strip().split("-"))
        res.append((a, b))

    res.sort()
    return res


def find_allowed_IP_ranges(blacklist, lower: int, upper: int, verbose=False):
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


def solve(data: str):
    blacklist = parse(data)
    upper = 9 if len(data) == 11 else 4294967295

    allowed_IP_ranges = find_allowed_IP_ranges(blacklist=blacklist, lower=0, upper=upper)
    star1 = min(a for a, _ in allowed_IP_ranges)
    print(f"The lowest allowed IP is {star1}.")

    star2 = sum(b - a + 1 for a, b in allowed_IP_ranges)
    print(f"There are {star2} allowed IP addresses.")

    return star1, star2


def main():
    year, day = 2016, 20
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve, extra_kwargs_parser="ignore")
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
