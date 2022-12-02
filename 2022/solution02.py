def read_input():
    with open("input02.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = [line.split() for line in s.strip().split("\n")]
    return res


def tonum(char):
    """Maps a move (ABC or XYZ) to a number 0, 1, 2, representing rock, paper, or scissors"""
    map_ = dict(zip("ABCXYZ", [0, 1, 2, 0, 1, 2]))
    return map_[char]


def get_points(opponent_move, player_move):
    opponent_move = tonum(opponent_move)
    player_move = tonum(player_move)
    res = 0
    res += player_move + 1

    draw = opponent_move == player_move
    won = player_move == (opponent_move + 1) % 3
    if draw:
        res += 3
    elif won:
        res += 6

    return res


def get_points2(opponent_move, outcome):
    opponent_move = tonum(opponent_move)
    res = 0
    draw = outcome == "Y"
    win = outcome == "Z"
    player_move = None
    if draw:
        res += 3
        player_move = opponent_move
    elif win:
        res += 6
        player_move = (opponent_move + 1) % 3
    else:
        player_move = (opponent_move + 2) % 3

    res += player_move + 1

    return res


def main():
    raw = read_input()
    parsed = parse(raw)

    scores = []
    scores2 = []
    for a, b in parsed:
        scores.append(get_points(a, b))
        scores2.append(get_points2(a, b))

    print(f"Total rock, paper, scissors points: {sum(scores)}.")

    print(f"Total points with correct guide: {sum(scores2)}.")


if __name__ == '__main__':
    main()
