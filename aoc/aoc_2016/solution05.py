import hashlib


def parse(s):
    res = s  # TODO parse input here
    return res


def parse(s):
    res = s.strip()
    return res


def crack_password(door_id, password_length=8):
    password = ["-" for _ in range(password_length)]
    char_ind = 0
    i = 0

    while char_ind < len(password):
        found_new_char = False
        key = door_id+str(i)
        hash_ = hashlib.md5(key.encode())
        hex_ = hash_.hexdigest()
        if hex_.startswith("00000"):
            password[char_ind] = hex_[5]
            char_ind += 1
            found_new_char = True
        i += 1

        if i % 1000 == 0 or found_new_char:
            print(f"Password: {''.join(password)}, hashing with {i}", end="\r")
        #

    print()
    return "".join(password)


def crack_password_fancy(door_id, password_length=8):
    password = ["-" for _ in range(password_length)]
    i = 0
    missing_chars = True

    while missing_chars:
        found_new_char = False
        key = door_id+str(i)
        hash_ = hashlib.md5(key.encode())
        hex_ = hash_.hexdigest()
        if hex_.startswith("00000"):
            try:
                pos = int(hex_[5])
                val = hex_[6]
                if password[pos] == "-":
                    password[pos] = val
                    found_new_char = True
                    missing_chars = any(c == "-" for c in password)
                #
            except (ValueError, IndexError):
                pass

        i += 1

        if i % 1000 == 0 or found_new_char:
            print(f"Password: {''.join(password)}, hashing with {i}", end="\r")
        #

    print()
    return "".join(password)


def solve(data: str):
    door_id = parse(data)

    star1 = crack_password(door_id=door_id)
    print(f"Solution to part 1: {star1}")

    print()
    star2 = crack_password_fancy(door_id=door_id)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2016, 5
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
