# ·`. ·*   `  ·.+  · • `.    * · ·  •  `. . ·*`      · `+  * ·· `  ·. +  ·. •`· 
# +·`.     · `+· *· .* · `· Scrambled Letters and Hash  · ` ·*  ·.  ·. `+  *· .`
# `.*·•·     *.`·    · https://adventofcode.com/2016/day/21   *·   * ··     `. *
# · ·  .*·`    +`·*. `• ·  .`·* ` ·   .+      · *·`.   `.+      · ·.* ` ` ·. +*·

import re


def parse(s: str):
    res = []
    patterns = {
        "swap_letter": r"swap letter (\D+) with letter (\D+)",
        "swap_position": r"swap position (\d+) with position (\d+)",
        "rotate_letter": r"rotate based on position of letter (\D+)",
        "rotate": r"rotate (\D+) (\d+) [step|steps]",
        "move": r"move position (\d+) to position (\d+)",
        "reverse": r"reverse positions (\d+) through (\d+)"
    }
    for line in s.split("\n"):
        for k, pat in patterns.items():
            m = re.match(pat, line)
            if m is not None:
                # Cast values as ints if possible
                arglist: list[int|str] = []
                for s in m.groups():
                    try:
                        arglist.append(int(s))
                    except ValueError:
                        arglist.append(s)
                    #

                tup = (k, arglist)
                res.append(tup)
                break
            #
        else:
            raise ValueError(f"Couldn't parse this line: {line}.")

    return res


class Scrambler:
    def __init__(self, instructions: list):
        self.instructions = [tup for tup in instructions]

    @staticmethod
    def swap_position(arr: list, ind1: int, ind2: int):
        arr[ind1], arr[ind2] = arr[ind2], arr[ind1]
        return arr

    def swap_letter(self, arr: list, let1: str, let2: str):
        inds = [arr.index(let) for let in (let1, let2)]
        return self.swap_position(arr, *inds)

    def rotate_letter(self, arr: list, let: str):
        ind = arr.index(let)
        steps = 1 + ind + int(ind >= 4)
        
        res = self.rotate(arr, "right", steps)
        return res

    def rotate(self, arr: list, direction: str, steps: int):
        cut = None
        steps %= len(arr)
        if direction == "left":
            cut = steps
        elif direction == "right":
            cut = len(arr) - steps
        else:
            raise ValueError

        arr = arr[cut:] + arr[:cut]

        return arr

    def move(self, arr: list, ind1: int, ind2: int):
        char = arr.pop(ind1)
        arr.insert(ind2, char)
        return arr

    @staticmethod
    def reverse(arr: list, ind1: int, ind2: int):
        arr[ind1:ind2+1] = arr[ind1:ind2+1][::-1]
        return arr

    def run_instruction(self, arr: list, instruction: tuple):
        operation, args = instruction
        fun = getattr(self, operation)
        res = fun(arr, *args)
        return res

    def __call__(self, input_: str):
        arr = list(input_)
        for instruction in self.instructions:
            arr = self.run_instruction(arr=arr, instruction=instruction)

        res = "".join(arr)
        return res


class Unscrambler(Scrambler):
    def __init__(self, instructions: list) -> None:
        # For the inverse scrambling, the inverse operations must be run in reverse order.
        self.instructions = [tup for tup in instructions[::-1]]
        self._reverse_ind_shift_lookup: dict[tuple[int, int], int] = dict()

    def rotate(self, arr: list, direction: str, steps: int):
        direction = {"left": "right", "right": "left"}[direction]
        return super().rotate(arr=arr, direction=direction, steps=steps)

    def move(self, arr: list, ind1: int, ind2: int):
        ind1, ind2 = ind2, ind1
        return super().move(arr=arr, ind1=ind1, ind2=ind2)

    def _reverse_ind_shift(self, i, n):
        """Ugly hack for figuring out which indices are shifted where during rotations."""
        k = (i, n)
        if k not in self._reverse_ind_shift_lookup:
            for ind in range(n):
                shifted = (ind + int(ind >= 4) + ind + 1) % n
                self._reverse_ind_shift_lookup[(shifted, n)] = ind
        res = self._reverse_ind_shift_lookup[k]
        return res

    def rotate_letter(self, arr: list, let: str):
        ind = arr.index(let)
        ind_original = self._reverse_ind_shift(ind, len(arr))
        shift = ind_original - ind

        res = self.rotate(arr, "left", shift)
        return res


def solve(data: str) -> tuple[int|str, int|str]:
    instructions = parse(data)

    scrambler = Scrambler(instructions=instructions)
    is_example = len(data) < 500
    password = 'abcde' if is_example else "abcdefgh"
    star1 = scrambler(password)
    print(f"The scrambled password is {star1}.")

    unscrambler = Unscrambler(instructions)
    scrambled_password = "fbgdceah"

    star2 = unscrambler(scrambled_password)
    print(f"Unscrambling the password '{scrambled_password}' results in: '{star2}'.")
    
    return star1, star2


def main() -> None:
    year, day = 2016, 21
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()