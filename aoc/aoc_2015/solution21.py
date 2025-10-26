# ·. *·  +. · ` . · `    ·  ·   ·+   ·   `  +··. *.`    ·   ·* ·+ `  . · .   ·`*
# *`· .·` *  ··.    • ·      .· RPG Simulator 20XX   ·.*   ·•`·      · .* ·  .· 
# ·· `     ·.• `·. • · https://adventofcode.com/2015/day/21 `   ·  ·. · · `   *·
# ` .·  ·      ·`*  · .· ·`      .··  *`· .•·+`   ··.    ·.   .  · *· .`   *· ·`


from copy import deepcopy
from itertools import combinations

def parse(s: str):
    res = {}
    for line in s.split("\n"):
        k, v = line.split(": ")
        res[k] = int(v)

    return res


store_contents = \
"""Weapons:    Cost  Damage  Armor
Dagger        8     4       0
Shortsword   10     5       0
Warhammer    25     6       0
Longsword    40     7       0
Greataxe     74     8       0

Armor:      Cost  Damage  Armor
Leather      13     0       1
Chainmail    31     0       2
Splintmail   53     0       3
Bandedmail   75     0       4
Platemail   102     0       5

Rings:      Cost  Damage  Armor
Damage +1    25     1       0
Damage +2    50     2       0
Damage +3   100     3       0
Defense +1   20     0       1
Defense +2   40     0       2
Defense +3   80     0       3"""


def parse_store():
    """Parses the store contents into nested dicts like
    {"Armor": {'armor 1': {'Cost': x, 'Damage': x}}} etc"""

    parts = store_contents.split("\n\n")
    res = {}
    for part in parts:
        lines = part.split("\n")
        cells = [[elem.strip() for elem in line.split("  ") if elem.strip()] for line in lines]
        headers = cells[0]
        category = headers[0][:-1]
        properties = headers[1:]

        d = {}
        for stuff in cells[1:]:
            item = stuff[0]
            d[item] = dict(zip(properties, map(int, stuff[1:])))
        res[category] = d

    return res


gear_properties = parse_store()


def get_possible_gear_combinations(store_contents, allowed):
    """Returns all possible gear combinations [{'Armor': (somearmor,), 'Weapon': etc}]"""
    all_combinations_by_type = {}
    for type_, n_allowed in allowed.items():
        # The items of the current type which we may buy
        options = list(store_contents[type_].keys())
        combs = sum([list(combinations(options, i)) for i in n_allowed], [])
        non_repeated = sorted({tuple(sorted(comb)) for comb in combs if len(comb) == len(set(comb))})
        all_combinations_by_type[type_] = non_repeated

    # Find the powerset of gear combinations
    res = []
    types = sorted(all_combinations_by_type.keys())
    for type_ in types:
        combs = all_combinations_by_type[type_]
        updated = []
        if not res:
            res = [{type_: comb} for comb in combs]
            continue
        for comb in combs:
            for d in res:
                new = {k: v for k, v in d.items()}
                new[type_] = comb
                updated.append(new)
            #
        res = updated
    return res


def aggregate_gear_properties(gear):
    """Sums the cost and damage/armor bonuses of a combination of gear, specified as
    {'Rings': (ring1, ring2, ...), ...}"""
    res = {}

    for type_, items in gear.items():
        for item in items:
            item_properties = gear_properties[type_][item]
            for prop, val in item_properties.items():
                res[prop] = res.get(prop, 0) + val
            #
        #
    return res


def player_stats():
    res = {
        "Hit Points": 100,
        "Damage": 0,
        "Armor": 0
    }

    return res


def compute_damage(damage, armor):
    res = max(1, damage - armor)
    return res


def player_wins(player_stats, boss_stats):
    """Determines whether the player wins the boss fight"""
    player_stats = deepcopy(player_stats)
    boss_stats = deepcopy(boss_stats)

    stats = [player_stats, boss_stats]
    n_its = 0
    while all(d["Hit Points"] > 0 for d in stats):
        this, other = stats[n_its % len(stats)], stats[(n_its + 1) % len(stats)]
        damage_dealt = compute_damage(damage=this["Damage"], armor=other["Armor"])
        other["Hit Points"] -= damage_dealt

        n_its += 1

    return player_stats["Hit Points"] > 0


# Constraints for gear (can have no armor or one armor, 0, 1, or 2 rings, etc)
allowed = {
    "Armor": (0, 1),
    "Rings": (0, 1, 2),
    "Weapons": (1,)
}


def get_gear_combinations_by_outcome(data_boss, must_win=True):

    gear_combos = get_possible_gear_combinations(gear_properties, allowed)

    res = []
    for combo in gear_combos:
        gear_stats = aggregate_gear_properties(combo)

        # Add gear bonus to player stats
        data_player = player_stats()
        for prop, bonus in gear_stats.items():
            if prop in data_player:
                data_player[prop] += bonus

        win = player_wins(data_player, data_boss)
        correct_outcome = win and must_win or not win and not must_win
        if correct_outcome:
            res.append(combo)
        #
    return res


def solve(data: str) -> tuple[int|str, int|str]:
    data_boss = parse(data)

    winning_combos = get_gear_combinations_by_outcome(data_boss=data_boss)
    star1 = min(aggregate_gear_properties(gear)["Cost"] for gear in winning_combos)
    print(f"Solution to part 1: {star1}")

    losing_combos = get_gear_combinations_by_outcome(data_boss=data_boss, must_win=False)
    star2 = max(aggregate_gear_properties(gear)["Cost"] for gear in losing_combos)

    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 21
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()