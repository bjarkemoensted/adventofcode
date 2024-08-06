class Parser:
    def __init__(self, verbose=False):
        # Bools to keep track of whether the parser is in 'ignore mode' (after '!') or 'garbage mode' (inside '< >')
        self.ignore = False
        self.garbage = False

        self.verbose = verbose

        # Keep track of the total of the group scores, and the amount of garbage encountered
        self.total_score = 0
        self.garbage_size = 0

        # Keep a history of all characters encountered, and when the current group was opened
        self._history = []
        self._opened_at = []

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _parse_char(self, char):
        self._history.append(char)

        # Handle escape characters (always ignore if previous character was '!', enter ignore mode on '!')
        if self.ignore:
            self.ignore = False
            return
        if char == '!':
            self.ignore = True
            return

        # Handle garbage groups (start on '<' stay in garbage mode until unescaped '>')
        if self.garbage:
            if char == '>':
                self.garbage = False
            else:
                self.garbage_size += 1
            return
        if char == "<":
            self.garbage = True
            return

        # Handle group matching (opened/closed by '{' and '}', unless in ignore/garbage mode)
        if char == '{':
            # Note the index where the group opened, so we can grab the entire group later on
            self._opened_at.append(len(self._history) - 1)
            return
        if char == '}':
            # Update the total score (add the depth of the group)
            score = len(self._opened_at)
            self.total_score += score

            # Optionally print group and score, for debugging and stuff
            group = ''.join(self._history[self._opened_at.pop():])
            self.vprint(f"Group {group} closed with score {score}: {''.join(self._history[-10:])}")
            return

    def parse(self, s: str):
        for char in s:
            self._parse_char(char)


def solve(data: str):
    parser = Parser()
    parser.parse(data)

    star1 = parser.total_score
    print(f"Solution to part 1: {star1}")

    star2 = parser.garbage_size
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 9
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
