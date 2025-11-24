# ·`  . *·  `. +·. `  +·*.· . *·` . ·* · .  `·   + ··`. +  .* ·  ·`    ·`*.·+`·*
# `·•  .   ·•·. ` *·`    If You Give A Seed A Fertilizer .* ·   ·`.  . *··`  . ·
# .*·.· `  .· +·* .    https://adventofcode.com/2023/day/5   *·`+  ·  .`  + ··* 
# *·.·  ·`* . ·  *   ·    *.·`  .•··    `. ·•`* ·   .··  ·.*   ·`.   ·*`  ·*` ..


from typing import TypeAlias

maptype: TypeAlias = list[list[int]]


def parse(s: str) -> tuple[list[int], list[maptype]]:
    """Parses to seeds and maps"""
    parts = s.split("\n\n")
    seeds: list[int] = [int(elem) for elem in parts[0].split(":")[1].strip().split(" ")]

    maps: list[maptype] = []
    for part in parts[1:]:
        thismap: maptype = []
        lines = part.split("\n")
        for line in lines[1:]:
            thismap.append([int(elem) for elem in line.strip().split(" ")])
        maps.append(thismap)
    
    return seeds, maps


def get_next(key: int, map_: maptype) -> int:
    """Takes a number, e.g. a seed number, and a map. Returns what the number maps to.
    If not match is found in the maps, the key is mapped to itself."""
    
    res = key
    for dest_start, source_start, range_len in map_:
        offset = key - source_start
        match = 0 <= offset < range_len
        if match:
            res = dest_start + offset
            break
        #
    return res


def get_path(seed: int, maps: list[maptype]) -> list[int]:
    """Takes a starting number (seed number) and applies all the maps, to end up at the corresponding
    location number"""
    route = [seed]
    running = seed
    for map_ in maps:
        running = get_next(running, map_)
        route.append(running)
    return route


def get_destination_intervals(source_interval: tuple[int, int], mappings: maptype):
    """Takes an interval (low, high) and a list of mappings.
    Returns a list of the intervals mapped to by the input interval.
    For example source_interval=(1, 10) and mappings = [(12, 5, 3)] would give
    [(1, 5), (12, 15), (9, 10)] because only the mapping is only applied
    to the sub-interval (5, 8), resulting in (12, 15).
    The remainder of the interval is mapped to itself.
    This is equivalent to mapping each individual element in the interval, but much more efficient."""

    # Start with the low and high limits of the source interval
    low, high = source_interval
    res = []

    # Order the mappings by the lower limit of the intervals to which they apply
    for mapping in sorted(mappings, key=lambda t: t[1]):
        # Break mapping into the lower limits of where the mapping goes to and from, respectively, and the interval size
        dest_low, domain_low, n = mapping
        shift = dest_low - domain_low
        domain_high = domain_low + n

        # If there's no overlap between mapping and source interval, there's nothing to do
        miss = (low >= domain_high) or (high < domain_low)
        if miss:
            continue

        # Any values below the mapping range are mapped to themselves
        if low < domain_low:
            # use identity mapping for the lower part of source
            res.append((low, domain_low))
            low = domain_low

        # The overlap between the source and the mapping is then mapped to the destination range
        newhigh = min(high, domain_high)
        res.append((low + shift, newhigh + shift))

        low = newhigh

    # If any values remain after applying all maps, they map to themselves
    if low < high:
        res.append((low, high))

    return res


def build_paths(start_nodes: list[tuple[int, int]], maps: list[maptype]) -> list[list[tuple[int, int]]]:
    """Takes a list of intervals e.g. [(79, 93), (55, 68)] and a list of mappings.
    The mappings are sequentially applied to produce all possible paths from intervals of starting (seed)
    values to intervals of location values."""

    # To start, each path is just the start node
    paths = [[n] for n in start_nodes]
    for maplist in maps:
        new_paths = []
        # Grow each path by the intervals mapped to (seed numbers to soild numbers, etc)
        for path in paths:
            endnode = path[-1]
            for newnode in get_destination_intervals(endnode, maplist):
                new_path = path + [newnode]
                new_paths.append(new_path)
            #
        paths = new_paths
    
    return paths


def solve(data: str) -> tuple[int|str, ...]:
    seeds, maps = parse(data)

    paths = [get_path(seed, maps) for seed in seeds]
    star1 = min(path[-1] for path in paths)
    print(f"Solution to part 1: {star1}")

    seed_intervals = [(seeds[i], seeds[i] + seeds[i + 1]) for i in range(0, len(seeds), 2)]

    paths2 = build_paths(start_nodes=seed_intervals, maps=maps)
    star2 = min([p[-1][0] for p in paths2])
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 5
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
