#  ·`·`  * . · .+ ·    ·.`*+   `·`*.   ·   ·.*  •`·.*· · ` .  .·* ` +.·   `. · *
# .* ·*.··+ `* ··  .+*   · ` An Elephant Named Joseph . · *    +  · .   *·.· ` .
# *·  .   · *   `. *·. https://adventofcode.com/2016/day/19 . *.·+   `   .·`·* ·
# · *.·`*.• ·`   * · · *`.  ·*    `*·   `..·*   . *·` ·+ .  ·`    · ·*  ·`* `·•·


import math


def parse(s: str):
    res = int(s)
    return res


def simulate_gift_game(n_participants):
    """Simulates the gift game where each elf takes their neighbors gifts.
    In each round, every other elf is eliminated due to having their gifts stolen.
    If there's an odd number of elves playing in a given round, the first elf is also eliminated after the round.
    For example, if 5 elves are playing, elves 1, 3, and 5, steal presents from elves 2, 4, and 1, so elf number 1
    also doesn't make it to the subsequent round."""

    participants = [i+1 for i in range(n_participants)]
    while len(participants) > 1:
        offset = len(participants) % 2
        next_participants = [participants[i] for i in range(offset, len(participants)) if i % 2 == 0]
        participants = next_participants

    winner = participants[0]
    return winner


def simulate_new_gift_game(n_participants):
    """Simulates the circular gift game, where elves steal gifts from the elves sitting directly across from them.
    Instead of tracking 'rounds', we continually update the participant list, so the first elf to play is always at
    index zero."""

    participants = [i+1 for i in range(n_participants)]

    while len(participants) > 1:
        if len(participants) % 2 != 0:
            # With an odd number of players, let the first elf steal a gift, and move them to the back of the list
            other_ind = math.floor(len(participants) / 2) % len(participants)
            del participants[other_ind]
            participants.append(participants.pop(0))
        else:
            # The elves in the latter half of the list are in danger of getting their gifts stolen.
            middle_ind = len(participants) // 2
            last_part = participants[middle_ind:]
            # Every third one keeps their gifts, however, due to the circle shrinking
            survivors = [last_part[i] for i in range(len(last_part)) if i % 3 == 2]

            # Compute how many elves just stole gifts. Move those to the back.
            n_eliminated = len(last_part) - len(survivors)
            # The remainder of the first 'safe' half, have yet to steal gifts before the 'survivors' from before.
            participants = participants[n_eliminated:middle_ind] + survivors + participants[:n_eliminated]

    winner = participants[0]
    return winner


def solve(data: str) -> tuple[int|str, int|str]:
    n_participants = parse(data)

    star1 = simulate_gift_game(n_participants)
    print(f"Elf number {star1} ends up with all the presents.")

    star2 = simulate_new_gift_game(n_participants)
    print(f"In the circular gift game, elf number {star2} ends up with all the presents.")

    return star1, star2


def main() -> None:
    year, day = 2016, 19
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
