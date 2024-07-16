def read_input():
    with open("input06.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = s.strip()
    return res


def find_package_start_location(s, marker_size=4):
    for i in range(len(s) - marker_size):
        snippet = s[i:i+marker_size]
        if len(set(snippet)) == marker_size:
            return i + marker_size


def main():
    raw = read_input()
    parsed = parse(raw)

    loc = find_package_start_location(parsed)
    print(f"First package start marker is at position {loc}.")

    loc2 = find_package_start_location(parsed, marker_size=14)
    print(f"First state-of-message packet is at position {loc2}.")

if __name__ == '__main__':
    main()
