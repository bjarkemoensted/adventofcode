#    *ꞏ. `. ⸳+   *ꞏ  •⸳.  .ꞏ+.          ꞏ`ꞏ*   ⸳.    + ⸳ .  • ꞏ. •⸳* .  `ꞏ  +⸳. 
# `ꞏ  .*+.ꞏ     ⸳ `    •ꞏ  Inventory Management System ꞏ •⸳ `       .++ ꞏ⸳   .ꞏ 
#   .ꞏ⸳  • `    +.  ꞏ⸳ https://adventofcode.com/2018/day/2 .ꞏ+   . ⸳⸳ .ꞏ  *ꞏ` •.
#  +⸳.* ꞏ  *⸳ꞏ.`ꞏ    +  . ⸳`.  ⸳ ꞏ `•* .         ⸳+ ꞏ ꞏ.   •   `.   ꞏ `⸳ .⸳+.ꞏ •

from collections import Counter


def parse(s):
    res = s.splitlines()
    return res


def compute_checksum(ids: list[str]) -> int:
    """Determines the checksum of the box IDs"""
    
    counts = [Counter(s) for s in ids]
    n_matches = (2, 3)
    res = 1
    # Get the product of the number of box IDs with symbols occurring 2 and 3 times    
    for n in n_matches:
        res *= sum(any(v==n for v in c.values()) for c in counts)
    return res


def get_prototype_id(ids: list[str]) -> str:
    """Determine the pair of IDs which differ in exactly one place.
    Return the ID resulting from keeping the matching symbols only"""
    for i, id_ in enumerate(ids):
        for other in ids[i+1:]:
            diff = [a != b for a, b in zip(id_, other, strict=True)]
            if sum(diff) == 1:
                res = "".join([id_[ind] for ind, d in enumerate(diff) if not d])
                return res
            #
        #
    raise ValueError


def solve(data: str):
    box_ids = parse(data)

    star1 = compute_checksum(box_ids)
    print(f"Solution to part 1: {star1}")

    star2 = get_prototype_id(box_ids)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2018, 2
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
