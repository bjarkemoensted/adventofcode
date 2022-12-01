def read_input():
    with open("input01.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    snippets = s.split("\n\n")
    calories = [[int(substring) for substring in snippet.split("\n")] for snippet in snippets]
    return calories


def main():
    raw = read_input()
    calories = parse(raw)

    totals = [sum(arr) for arr in calories]
    totals.sort(reverse=True)
    max_calories = totals[0]
    print(f"Max calories is: {max_calories}.")

    top_three_totals = sum(totals[:3])
    print(f"The 3 snack-heaviest elves carry {top_three_totals} calories.")


if __name__ == '__main__':
    main()
