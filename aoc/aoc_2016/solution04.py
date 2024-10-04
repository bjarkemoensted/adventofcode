from collections import Counter
import re
import string


def read_input():
    with open("input04.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = []
    for line in s.strip().split("\n"):
        # parse into a tuple of (text stuff, section ID, checksum)
        m = re.match(r"(.*)-(.*)\[(.*)\]", line)
        room = tuple(int(m.group(i)) if i == 2 else m.group(i) for i in range(1, 4))
        res.append(room)

    return res


def compute_checksum(code):
    counts = Counter(code.replace("-", ""))
    correct_checksum = "".join(sorted(counts.keys(), key=lambda char: (-counts[char], char))[:5])
    return correct_checksum


def decrypt_room_name(code, id_):
    alph = string.ascii_lowercase
    lookup = {let: i for i, let in enumerate(alph)}
    res = ""
    for char in code:
        if char == "-":
            newchar = " "
        else:
            startind = lookup[char]
            newchar = alph[(startind + id_) % len(alph)]
        res += newchar

    return res


def main():
    raw = read_input()
    parsed = parse(raw)

    real_rooms_ids = []
    decrypted_data = []

    for code, id_, checksum in parsed:
        correct_checksum = compute_checksum(code)
        room_is_real = correct_checksum == checksum
        if room_is_real:
            real_rooms_ids.append(id_)
            name_ = decrypt_room_name(code, id_)
            decrypted_data.append((name_, id_, checksum))

    print(f"Read room ids sum to {sum(real_rooms_ids)}.")

    north_pole_room = [t for t in decrypted_data if "north" in t[0]]
    assert len(north_pole_room) == 1
    target_id = north_pole_room[0][1]

    print(f"North pole objects are stored in sector {target_id}.")


if __name__ == '__main__':
    main()
