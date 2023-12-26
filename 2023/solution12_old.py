from functools import cache


test = """???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1"""









def count_combinations_old(groups, code):
    res = 0
    s = ".".join(groups)
    for newstring in all_replacements(s):
        if tuple(len(elem) for elem in split(newstring)) == code:
            res += 1

    return res







###############################

def read_input():
    with open("input12.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s, unfold=False):
    res = []
    for line in s.split("\n"):
        a, b = line.split()
        if unfold:
            a = "?".join([a for _ in range(5)])
        nums = tuple(int(elem) for elem in b.split(","))
        if unfold:
            nums = tuple(5*list(nums))
        res.append((a, nums))

    return res


def _all_replacements(s):
    """Generates all the possible strings resulting from replacements of "?" with '.' or "#"."""

    if "?" in s:
        for char in ".#":
            yield from _all_replacements(s.replace("?", char, 1))
        #
    else:
        yield s


@cache
def get_replacements_with_multiplicities(s):
    counts = {}
    for newstring in _all_replacements(s):
        k = newstring.strip(".")
        try:
            counts[k] += 1
        except KeyError:
            counts[k] = 1
        #
    res = counts.items()
    return res


def count_combinations(springs: str, code: tuple, depth=0, verbose=False):
    nextkwargs = dict(depth=depth+1, verbose=verbose)
    def vprint(*args, **kwargs):
        if verbose:
            print(2*depth*"*", *args, **kwargs)
    springs = str(springs).strip(".")

    vprint(f"Called with {springs}, {code}")

    if not springs and not code:
        return 1
    elif not springs or not code and "#" in springs:
        return 0

    res = 0
    if springs.endswith("#"):
        n_needed = code[-1]
        too_short = len(springs) < n_needed or "." in springs[-n_needed:]
        too_long = len(springs) > n_needed and springs[-n_needed-1] == "#"
        if too_long or too_short:
            return 0
        newcode = tuple(code[:-1])
        cut = n_needed + 1 if len(springs) > n_needed else n_needed
        newsprings = springs[:-cut]

        res += count_combinations(newsprings, newcode, **nextkwargs)
    else:
        cut = springs.rfind(".")
        if cut > -1:
            pre_split = springs[:cut+1]
            post_split = springs[cut+1:]
        else:
            pre_split = springs[:-1]
            post_split = springs[-1]
        for new_comb, mul in get_replacements_with_multiplicities(post_split):
            newsprings = pre_split + new_comb
            res += mul*count_combinations(newsprings, code, **nextkwargs)


    #print(springs, code)

    return res



def main():
    raw = read_input()
    data = parse(raw)

    #star1 = sum(count_combinations(groups, code) for groups, code in data)
    star1 = 0
    for i, (springs, code) in enumerate(data):
        a = count_combinations(springs, code)
        star1 += a
    print(f"Number of combinations sum to {star1}.")

    data_unfolded = parse(raw, unfold=True)
    star2 = 0
    for i, (springs, code) in enumerate(data_unfolded):
        print(springs, code)
        star2 += count_combinations(springs, code)
        print(f"Processed {i+1} of {len(data_unfolded)} ({(i+1)*100/len(data_unfolded):.1f}%)")


    #print(f"Afgter unfolding the instructions, the number of combinations sum to {star2}.")


if __name__ == '__main__':
    main()
