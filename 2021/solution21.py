with open("input21.txt") as f:
    starting_positions = [int(line.split("position: ")[-1]) for line in f.readlines()]


def multiroll(rolls, n_throws=3):
    throws = []
    while True:
        for roll in rolls:
            throws.append(roll)
            if len(throws) >= n_throws:
                yield throws
                throws = []
            #
    #


def play_game(rolls, player_positions, n_throws=3):
    player_positions = [val for val in player_positions]
    n = 0
    player_turn = -1
    points = [0 for _ in player_positions]
    cumrolls = multiroll(rolls, n_throws)

    while all(point < 1000 for point in points):
        player_turn = (player_turn + 1) % len(player_positions)
        roll = next(cumrolls)
        roll_sum = sum(roll)
        n += n_throws
        new_pos = 1 + ((player_positions[player_turn] + roll_sum - 1) % 10)
        player_positions[player_turn] = new_pos
        points[player_turn] += new_pos

        #print(f"Roll={roll_sum} ({'+'.join(map(str, roll))}). Player {player_turn + 1} now has {points[player_turn]} points.")

    loser_points = min(points)
    num = loser_points*n
    return num


star1 = play_game(rolls=range(1, 101), player_positions=starting_positions)
print(f"Solution to star 1: {star1}.")