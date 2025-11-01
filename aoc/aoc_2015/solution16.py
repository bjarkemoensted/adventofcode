# .·.*· ·`  ·.  *   ·`.   .·*.`    ·   ·   . · • * .· · + `·  *  .+`   .·· *`.· 
# `.*·`. •   * ` ·  .*· ·.*`·   .    Aunt Sue  ·. .+  `·  + ·    • .` ·  *.   .·
# ·*  .` ·    .*·  ·   https://adventofcode.com/2015/day/16   .·`·   •   .`·  +`
# •.·`*·. ·+·  ·.• . ·+·* `    .*··.  *.` +·`·     ·  * .·  + ·.   `*·  .   ..`*


def parse(s: str):
    res = {}
    for line in s.split("\n"):
        sue = int(line.split(":")[0][4:])
        snip = ": ".join(line.split(": ")[1:])
        data = {}
        for part in snip.split(", "):
            k, v = part.split(": ")
            data[k] = int(v)

        res[sue] = data

    return res


def determine_sue(d, forensics):
    # Makes list of all auntie Sues and eliminate those for whom the input is inconsistent with evidence
    candidate_sues = sorted(d.keys())
    for i in range(len(candidate_sues) - 1, -1, -1):
        sue = candidate_sues[i]
        if any(forensics[k] != v for k, v in d[sue].items()):
            del candidate_sues[i]

    assert len(candidate_sues) == 1
    res = candidate_sues[0]
    return res


def determine_sue2(d, forensics):
    # Go again but with the updated criteria
    candidate_sues = sorted(d.keys())
    for i in range(len(candidate_sues)-1, -1, -1):
        sue = candidate_sues[i]
        eliminate = False
        for k, v in d[sue].items():
            expected = forensics[k]
            if k in ('cats', 'trees'):
                if v <= expected:
                    eliminate = True
                #
            elif k in ('pomeranians', 'goldfish'):
                if v >= expected:
                    eliminate = True
                #
            elif v != expected:
                eliminate = True
        if eliminate:
            del candidate_sues[i]
        #

    res = candidate_sues[0]
    return res


def solve(data: str) -> tuple[int|str, int|str]:
    d = parse(data)

    forensic_analysis_str = \
        """children: 3
        cats: 7
        samoyeds: 2
        pomeranians: 3
        akitas: 0
        vizslas: 0
        goldfish: 5
        trees: 3
        cars: 2
        perfumes: 1"""

    forensics = {c.strip(): int(q) for c, q in map(lambda s: s.split(": "), forensic_analysis_str.split("\n"))}

    star1 = determine_sue(d, forensics)
    print(f"Solution to part 1: {star1}")

    star2 = determine_sue2(d, forensics)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 16
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
