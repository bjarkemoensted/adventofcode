# Read in data
with open("input01.txt") as f:
    data = [int(line) for line in f.readlines()]


# Star 1
n_increased = sum(data[i+1] > data[i] for i in range(len(data) - 1))
print(f"First puzzle: {n_increased}")

# Star 2
n_increased_running = sum(data[i+3] > data[i] for i in range(len(data) - 3))
print(f"Second puzzle: {n_increased_running}")
