# ·.·  ` · .*··   ·`*   . ·` * · * .·  ·`  . *  · ·` .` · +   · *. · *` +.·.` ·•
# ·` ·*.   ·  *· + ·`·. *   · •` Binary Boarding `·.  * +·    • · ..+··` * `· +·
# .·. · *  +·+ `       https://adventofcode.com/2020/day/5 `··*  ·  .• · ` *  ·`
# *·`. ·  +  ` · ·   .··`   *· .· *·`*·  .·*` ·+ .`  * ·• ·. `  *+· ·` . ·•·+·`·


def parse(s: str) -> list[tuple[int, int]]:
    """Returns a list of the (i, j) coordinates of each seat"""
    res: list[tuple[int, int]] = []
    for seat_code in s.strip().splitlines():
        rowcode = seat_code[:7]
        binary = rowcode.replace("F", "0").replace("B", "1")
        i = int(binary, 2)
        
        colcode = seat_code[-3:]
        binary = colcode.replace("L", "0").replace("R", "1")
        j = int(binary, 2)
        res.append((i, j))
        
    return res


def get_seat_id(coord: tuple[int, int]) -> int:
    """Returns the ID of a given seat"""
    i, j = coord
    res = i*8 + j
    return res


def determine_remaining_seat_id(seat_coords: list[tuple[int, int]]) -> int:
    all_IDs = {get_seat_id(coord) for coord in seat_coords}

    for i in range(128):
        for j in range(8):
            sid = get_seat_id((i, j))

            seat_is_free = sid not in all_IDs
            neighbours_exist = all(val+sid in all_IDs for val in (1, -1))
            correct = seat_is_free and neighbours_exist
            if correct:
                return sid
            #
        #
    
    raise RuntimeError


def solve(data: str) -> tuple[int|str, ...]:
    seat_coords = parse(data)
    seat_ids = [get_seat_id(coord) for coord in seat_coords]

    star1 = max(seat_ids)
    print(f"Solution to part 1: {star1}")

    star2 = determine_remaining_seat_id(seat_coords)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 5
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
