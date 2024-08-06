# Read in data
with open("input14.txt") as f:
    puzzle_input = f.read()

example_input = \
"""Comet can fly 14 km/s for 10 seconds, but then must rest for 127 seconds.
Dancer can fly 16 km/s for 11 seconds, but then must rest for 162 seconds."""


def parse(s):
    """Parses reindeer info. Gives a dict mapping reindeer names to properties"""
    res = {}
    for line in s.split("\n"):
        words = line.split(" ")
        name = words[0]
        speed = int(words[3])
        run_time = int(words[6])
        rest_time = int(words[-2])
        res[name] = dict(speed=speed, run_time=run_time, rest_time=rest_time)

    return res


def determine_distance_travelled(reindeer, n_seconds):
    # Figure out the number of full rest/run cycles the reindeer can do, and the corresponding distance
    distance_in_one_cycle = reindeer["speed"]*reindeer["run_time"]
    cycle_time = reindeer["run_time"] + reindeer["rest_time"]
    n_cycles = n_seconds // cycle_time
    distance = n_cycles * distance_in_one_cycle

    # Add the distance travelled in the final, partial cycle
    remaining_runtime = min(reindeer["run_time"], n_seconds%cycle_time)
    distance += remaining_runtime*reindeer["speed"]

    return distance


data = parse(puzzle_input)

n_seconds_for_race = 2503
name2dist = {name: determine_distance_travelled(d, n_seconds_for_race) for name, d in data.items()}

print(f"Fastest reindeer travelled {max(name2dist.values())} km.")


def compute_points(reindeers, n_seconds):
    points = {k: 0 for k in reindeers.keys()}
    positions = {k: 0 for k in reindeers.keys()}

    for second in range(n_seconds):
        for reindeer, d in reindeers.items():
            # Increment reindeer position if it's currently running
            cycle_time = d["run_time"] + d["rest_time"]
            running = second % cycle_time < d["run_time"]
            if running:
                positions[reindeer] += d["speed"]
            #
        # Award a point to reindeers who currently hold the lead position
        lead_pos = max(positions.values())
        for reindeer, pos in positions.items():
            if pos == lead_pos:
                points[reindeer] += 1
            #
        #
    return points


race_results = compute_points(data, n_seconds_for_race)
print(f"Winner of second reindeer race has {max(race_results.values())} points.")
