from collections import Counter

with open("input06.txt") as f:
    fish_countdowns = [int(x) for x in f.read().strip().split(",")]
    fish_data = dict(Counter(fish_countdowns))


def run_simulation(fish, n_iterations=80):
    fish = {k: v for k, v in fish.items()}

    for i in range(n_iterations):
        # Number of new fish spawned in this round
        n_new = fish.get(0, 0)

        # Decrement the counter until new offspring for all fish
        new_data = {i: 0 for i in range(9)}
        for k, v in fish.items():
            # If fish spawn (timer=0), timer resets to 6. Else decrease by 1.
            new_key = 6 if k == 0 else k - 1
            new_data[new_key] += v

        # Add new fish to state
        new_data[8] += n_new
        fish = new_data

    return sum(fish.values())


n_fish_80_days = run_simulation(fish_data)
print(f"Solution to star 1: {n_fish_80_days}.")

n_fish_256_days = run_simulation(fish_data, n_iterations=256)
print(f"Solution to star 2: {n_fish_256_days}.")
