import string


def read_input():
    with open("input03.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = [line for line in s.strip().split("\n")]
    return res


def get_compartment_overlap(s):
    cut = len(s) // 2
    a, b = s[:cut], s[cut:]
    assert len(a) == len(b)

    overlap = set(a).intersection(set(b))
    assert len(overlap) == 1

    res = list(overlap)[0]
    return res


def get_priority(char):
    letters = string.ascii_lowercase + string.ascii_uppercase
    d = {letter: i+1 for i, letter in enumerate(letters)}

    res = d[char]
    return res


def main():
    raw = read_input()
    parsed = parse(raw)

    priorities = []

    for line in parsed:
        overlap = get_compartment_overlap(line)
        priority = get_priority(overlap)
        priorities.append(priority)

    print(f"Priorities sum to {sum(priorities)}.")

    groups = [parsed[i:i+3] for i in range(0, len(parsed), 3)]

    badges = []
    for group in groups:
        items = set(group[0])
        for new_member in group[1:]:
            items = items.intersection(set(new_member))
        assert len(items) == 1
        badges.append(list(items)[0])

    badge_sum = sum(get_priority(badge) for badge in badges)
    print(f"Badges sum to {badge_sum}.")



if __name__ == '__main__':
    main()
