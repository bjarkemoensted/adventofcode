# . .·* · ·` + · •   * .·.   `+·    ·•·   `* ·..   ·    · ·+ `  ·.   · .+·`·. · 
#  ·  .`•·   ·     · ·  *+·   · Reindeer Olympics `. ·  * .· ·.·+     `· . `*· ·
# ·.*` ·   *·.    ·•`  https://adventofcode.com/2015/day/14  *· .+ ·` ·   ·  * .
# ``·     •·. ·+  *`·    *.  · *  `• . ·        ·`  ·.·•`.+ ·    ·.`·`* ·` *·`. 


def parse(s: str):
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


def solve(data: str) -> tuple[int|str, int|str]:
    reindeer_stats = parse(data)

    n_seconds_for_race = 2503
    name2dist = {name: determine_distance_travelled(d, n_seconds_for_race) for name, d in reindeer_stats.items()}

    star1 = max(name2dist.values())
    print(f"Solution to part 1: {star1}")

    race_results = compute_points(reindeer_stats, n_seconds_for_race)

    star2 = max(race_results.values())
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 14
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
