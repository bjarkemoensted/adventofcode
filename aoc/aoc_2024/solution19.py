# `. · +*`    ·* .   · `. · `*·   · .· *` · ·   ` · ·`*.  `· •·  ·*· `   *. ·*·`
#  * `··.   *·`·   ·`.·• . + ·` ·  Linen Layout  ·     ·` ·*·` .     + ·.  *`·.·
# ·`·*. `•· ·`   +  *  https://adventofcode.com/2024/day/19  ·  ·  * .· `  .·  *
# ··`.   ·`*· ..  · ·+•·*  `  .·`· .·`   + ·`*.   · `     ·    *. ·`·*.•  ··.`*·


def parse(s: str):
    towelpart, patternpart = s.split("\n\n")
    towels = towelpart.split(", ")
    patterns = patternpart.splitlines()
    return towels, patterns


class PatternCounter:
    """Contains functionality for taking an 'alphabet' (a collection of available strings),
    and a pattern (a target string), and computing the multiplicity of the pattern, i.e.
    the number of distinct ways the target string can be produced by the alphabet."""

    def __init__(self, alphabet):
        self.alphabet = tuple(alphabet)
        # Cache for computed results. Initialize with the base case
        self._counts = {"": 1}
    
    def __call__(self, pattern: str) -> int:
        """Returns the number of ways the alphabet can be arranged into the target pattern string.
        Works by comparing all strings in the alphabet with the start of the pattern,
        then recursing on the remainder of the string."""
        
        # Return cached result if available
        try:
            return self._counts[pattern]
        except KeyError:
            pass
        
        res = 0
        
        for substr in self.alphabet:
            if not pattern.startswith(substr):
                continue  # Skip substrings that don't match the beginning
            
            # Recurse on the remainder of the string and add counts to result
            n_chars = len(substr)
            next_pattern = pattern[n_chars:]
            res += self.__call__(pattern=next_pattern)
        
        # Add to cache before returning
        self._counts[pattern] = res
        
        return res
    #



def solve(data: str) -> tuple[int|str, int|str]:
    towels, patterns = parse(data)
    
    pc = PatternCounter(alphabet=towels)
    
    # A design is possible if its multiplicity is nonzero
    multiplicities = [pc(pattern) for pattern in patterns]
    star1 = sum(mul > 0 for mul in multiplicities)
    print(f"Solution to part 1: {star1}")

    # Just sum the multiplicities for each design in part 2
    star2 = sum(multiplicities)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 19
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
