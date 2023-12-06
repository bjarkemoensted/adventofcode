import math


def read_input():
    with open("input06.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = [[int(elem) for elem in line.split(":")[1].strip().split()] for line in s.split("\n")]
    return res


def solve_for_record(time_, record):
    sq = (time_**2 - 4*record)**0.5
    low = (time_ - sq)/2
    high = (time_ + sq) / 2
    return low, high


def n_ways_win(time_, record):
    low, high = solve_for_record(time_, record)
    n = math.floor(high) - math.ceil(low) + 1
    return n


def win_product(races):
    res = 1
    for time_, record in races:
        n = n_ways_win(time_, record)
        res *= n

    return res


def main():
    raw = read_input()
    timedist = parse(raw)
    races = list(zip(*timedist))

    star1 = win_product(races)
    print(f"Product of number of ways to win is {star1}.")

    bigrace = tuple(int("".join(map(str, arr))) for arr in timedist)
    star2 = n_ways_win(*bigrace)

    print(f"Number of ways to win the big race: {star2}.")


if __name__ == '__main__':
    main()

