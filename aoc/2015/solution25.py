# Read in data
with open("input25.txt") as f:
    puzzle_input = f.read()


def parse(s):
    s = s.replace(",", "").replace(".", "")
    words = s.split(" ")
    d = {words[i]: words[i+1] for i in range(len(words) - 1)}
    row = int(d["row"])
    col = int(d["column"])

    return row, col


def compute_next_code(old_code):
    m = 252533
    d = 33554393

    prod = old_code*m
    new_code = prod % d
    return new_code


def get_code_at_coords(initial, i, j):
    n_running = 1
    code_running = initial

    # The number of times we must iterate the code to reach the i, jth cell in the code manual
    n = sum(range(i + j - 1)) + j

    for _ in range(n-1):
        code_running = compute_next_code(code_running)
        n_running += 1
        if n_running % 1000 == 0:
            print(f"Progress = {100*n_running/n:.1f}%.", end="\r")
        #
    print()

    return code_running


coords = parse(puzzle_input)
initial_code = 20151125

star1 = get_code_at_coords(initial_code, *coords)
print(f"The correct code is: {star1}.")
