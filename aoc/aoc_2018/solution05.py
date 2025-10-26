# *·. `.· ·`* ·.  ·`*   ·.  ` ·.+*     ` · . · *`.·  .+` ·`    .+ · .*`·   . ·*.
# ·. · *.  +`·  * .· `·*.      Alchemical Reduction  · `· * `. ·   .      · *. ·
# · `   + . ·  `·.*.·  https://adventofcode.com/2018/day/5  ·•*` ·    · •.  ` · 
# .*· •`   · ` •.    ·`  +·.·•*·  . * · `  ·     · +. `·*   . ·•   · ` .·*.+  ·`


def react(units: list[str]) -> str:
    """Fully reacts a polymer by repeatedly eliminating neighboring pairs of letters where only
    the case differs"""
    
    # Map all letters to their reactant (same letter, opposite case)
    distinct_chars = {c.lower() for c in units}
    pairs = dict(sum([[(c.upper(), c), (c, c.upper())] for c in distinct_chars], []))
    
    # Go over the letters and handle any reactions encountered
    stack: list[str] = []
    for char in units:
        # If the newly encountered letter matches the stack, remove both
        match = stack and (stack[-1] == pairs[char])
        if match:
            stack.pop()
        else:
            # Add to the stack overwise
            stack.append(char)
        #
    
    res = "".join(stack)
    return res


def shortest_with_exclusion(units) -> int:
    """Determine the shortest possible polymer by removing one of the letters."""
    
    distinct_chars = {c.lower() for c in units}
    # Generate the alternative units we could work with by removing one letter (both upper and lower)
    alt = ([c for c in units if c.lower() != badchar] for badchar in distinct_chars)
    
    # Get the minimum length of the polymer resulting from the reactions
    reacted = map(react, alt)
    res = min(map(len, reacted))
    return res


def solve(data: str) -> tuple[int|str, int|str]:
    units = list(data)
    
    star1 = len(react(units))
    print(f"Solution to part 1: {star1}")

    star2 = shortest_with_exclusion(units)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 5
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()