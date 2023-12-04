import re


def read_input():
    with open("input04.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    games = []
    for line in s.split("\n"):
        m = re.match(r'Card\s*(\d+):\s*(.*)\|(.*)', line)
        matches = m.groups()
        _ = int(matches[0])
        winners, drawn = (tuple(int(elem) for elem in substring.strip().split()) for substring in matches[1:])
        games.append((winners, drawn))
    return games


def get_n_winning_numbers(winners, drawn):
    n = len(set(winners).intersection(set(drawn)))
    return n


def compute_points(winners, drawn):
    n = get_n_winning_numbers(winners, drawn)
    res = 0 if n == 0 else 2**(n - 1)
    return res


def count_cards(games):
    n_cards = [1 for _ in games]
    for i, (winners, drawn) in enumerate(games):
        n = get_n_winning_numbers(winners, drawn)
        for shift in range(i + 1, i + n + 1):
            try:
                n_cards[shift] += n_cards[i]
            except IndexError:
                pass
        #

    return n_cards


def main():
    raw = read_input()
    games = parse(raw)

    points = [compute_points(winners, drawn) for winners, drawn in games]
    star1 = sum(points)
    print(f"The elf won {star1} points.")

    n_cards = count_cards(games)
    star2 = sum(n_cards)
    print(f"Total number of cards: {star2}.")


if __name__ == '__main__':
    main()
