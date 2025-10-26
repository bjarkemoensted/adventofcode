# .·`    .·`+·   ` * .  ·  ·    `  .*· ·+`   *·.`  ·   · *  ` ·` · *    ` . · *·
# ·..·` *      `*· ·*`  .·`*·+`·.· Print Queue  .`   *·`+ .`  `· *  .·  *``  ·`*
# *` .•*·`.· ` *·   ·  https://adventofcode.com/2024/day/5  *· .·    `` ·*·. ` +
# `+`*·.    ·   `   `·.* `     ·  .` +· ·  `*    ·` .`* . `··*     · . `  *·` ·`

from collections import defaultdict


def parse(s: str):
    a, b = s.split("\n\n")
    
    # rules like X|Y meaning page X updates must come before page Y updates
    rules = defaultdict(lambda: set([]))
    for line in a.splitlines():
        k, v = map(int, line.split("|"))
        rules[k].add(v)
    
    rules = dict(rules)
    
    updates = [[int(part) for part in line.split(",")] for line in b.splitlines()]
    
    return rules, updates


def update_is_valid(update, rules):
    """Checks if an update is valid, i.e. it satisfies all provided rules.
    Iterates backwards through the list and checks that none of the preceeding numbers should
    come after the number at current ind."""
    
    for i in range(len(update)-1, -1, -1):
        prior = set(update[:i])
        
        # Get the rules for this page number
        num = update[i]
        num_must_be_before = rules.get(num, set([]))
        
        # If any rules are violated, the update is invalid
        violations = prior & num_must_be_before
        if violations:
            return False
        #
    
    return True


def _get_middle_elem(arr):
    """Grabs the middle element from a list. Throws error if list has even length"""
    assert len(arr) % 2 == 1
    return arr[len(arr) // 2]


def sum_middle_pages_if_valid(updates, rules):
    """Sums the page numbers in the middle of valid updates"""
    res = 0
    for update in updates:
        if update_is_valid(update, rules):
            res += _get_middle_elem(update)
        #
    
    return res


def sort(update, rules):
    """Sorts an update according to the specified rules.
    Works by identifying for each page number the number of constraints for that page number,
    i.e. mapping each page number to the page numbers which mustnot preceed it.
    Page numbers are then sorted according to the number of such constraints."""

    # Identify relevant constraints (only keep page numbers contained in the update)
    all_nums = set(update)
    constraints = {num: all_nums & rules.get(num, set([])) for num in update}
    
    res = sorted(update, key=lambda num: -len(constraints[num]))
    
    return res

def sort_and_sum(updates, rules):
    """Sorts invalid updates and returns the sum of their middle pages after sorting."""
    
    res = 0
    
    for upd in updates:
        # Ignore updates that are already valid
        if update_is_valid(update=upd, rules=rules):
            continue
        
        # Sort and check that update is valid after sorting
        sorted_upd = sort(upd, rules)
        assert update_is_valid(sorted_upd, rules)
        
        res += _get_middle_elem(sorted_upd)
    
    return res
    
    


def solve(data: str) -> tuple[int|str, int|str]:
    rules, updates = parse(data)

    star1 = sum_middle_pages_if_valid(updates=updates, rules=rules)
    print(f"Solution to part 1: {star1}")

    star2 = sort_and_sum(updates=updates, rules=rules)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 5
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()