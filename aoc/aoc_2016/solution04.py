# ꞏ• ⸳ +     ꞏ`•⸳  .⸳` ⸳+  `*ꞏ   ` +  ꞏ.  ⸳   ꞏ*` . ⸳ ` +.+ ꞏ ⸳.`  . ꞏ  .+  ꞏ ⸳•
# * .` `ꞏ     +.  *  ꞏ +    Security Through Obscurity   ꞏ +⸳..ꞏ  ⸳`ꞏ .  `+  ⸳ꞏ.
# ⸳ꞏ +.*⸳  ⸳` . *  `   https://adventofcode.com/2016/day/4 . ` + ⸳    *` ꞏ ⸳`.+ 
# ꞏ •. . ⸳ +  ꞏ⸳ .`+  . ꞏ`    `*ꞏ⸳+ . .ꞏ *   ꞏ   *.ꞏ  +ꞏ.`⸳*.ꞏ     •⸳.`      *`ꞏ


from collections import Counter
import re
import string


def parse(s):
    res = s  # TODO parse input here
    return res


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


def solve(data: str):
    parsed = parse(data)

    real_rooms_ids = []
    decrypted_data = []

    for code, id_, checksum in parsed:
        correct_checksum = compute_checksum(code)
        room_is_real = correct_checksum == checksum
        if room_is_real:
            real_rooms_ids.append(id_)
            name_ = decrypt_room_name(code, id_)
            decrypted_data.append((name_, id_, checksum))

    star1 = sum(real_rooms_ids)
    print(f"Solution to part 1: {star1}")

    
    north_pole_room = [t for t in decrypted_data if "north" in t[0]]
    star2 = north_pole_room[0][1] if len(north_pole_room) == 1 else None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2016, 4
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
