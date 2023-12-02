import re


def read_input():
    with open("input02.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    lines = s.split("\n")
    res = {}

    for line in lines:
        m = re.match(r"Game (\d+): (.*)", line)
        game_s, rest = m.groups()
        game_n = int(game_s)

        games = []
        for raw in rest.split("; "):
            game = {}
            for substring in raw.split(", "):
                n_s, color = substring.split(" ")
                game[color] = int(n_s)
            games.append(game)
        res[game_n] = games

    return res


def sample_is_possible(sample, population):
    res = all(population.get(color, 0) >= n_sample for color, n_sample in sample.items())
    return res


def game_is_possible(game: list, population):
    return all(sample_is_possible(sample, population) for sample in game)


def determine_min_cube_power(game, population):
    res = 1
    for color in population.keys():
        res *= max(sample.get(color, 0) for sample in game)
    return res


def main():
    raw = read_input()
    data = parse(raw)

    cubes = dict(red=12, green=13, blue=14)

    ids_of_possible_games = [id_ for id_, game in data.items() if game_is_possible(game, cubes)]
    star1 = sum(ids_of_possible_games)
    print(f"Sum of IDs of possible games: {star1}.")

    min_powers = [determine_min_cube_power(game, cubes) for game in data.values()]
    star2 = sum(min_powers)
    print(f"Sum of minimum cube powers is {star2}.")


if __name__ == '__main__':
    main()
