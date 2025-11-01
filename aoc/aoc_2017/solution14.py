# ··`.    ·.` ·*  + `   ·.  ·`·*. ·*  .  `*• ·     .· ·   *· `·+.·* `. · *.·`··+
#   .  *··` *.  · ·  .· `+   . Disk Defragmentation   .  .`* · ·      ·.  +`·. ·
# ·`·*   • ·.·* `·  +  https://adventofcode.com/2017/day/14 · *.     `• ·· *  .·
# `  ·· *.  · .` •*· ·. * ·`.·     •·+ ·.     *·    `   ·   *.· ` `+·*.     ·* .


def parse(s: str):
    res = s
    return res


hex_digit2binary = {hex(n)[-1]: bin(n)[2:].zfill(4) for n in range(16)}


def _emulate_knot(lengths, n: int = 256, n_repititions: int = 64):
    """Simulate the knot hashing thing."""
    pos = 0
    vals = list(range(n))
    skip_size = 0

    for _ in range(n_repititions):
        for len_ in lengths:
            # Wrapping is hard, so just make a temp list shifted by current position, then shift back later
            temp = vals[pos:] + vals[:pos]
            # Reverse the selected list (no problem with wrapping bc the selection starts at index 0 now)
            temp[:len_] = temp[:len_][::-1]
            # Shift back
            vals = temp[-pos:] + temp[:-pos]
            # Update position and skip size
            pos = (pos + len_ + skip_size) % n
            skip_size += 1

    return vals


def _densify(vals: list) -> str:
    """Makes a dense hex representation of a knot hash."""
    hex_parts = []
    block_size = 16

    for i in range(0, len(vals), block_size):
        # XOR each block together (0 ^ n always gives n)
        running = 0
        for val in vals[i: i+block_size]:
            running = running ^ val

        # Determine the two/digit hex representation of the block
        hex_ = hex(running)[2:]
        if len(hex_) == 1:
            hex_ = "0"+hex_
        hex_parts.append(hex_)

    res = "".join(hex_parts)
    return res


def knot_hash(s):
    lengths = [ord(char) for char in s]
    standard_length_suffixes = [17, 31, 73, 47, 23]
    lengths += standard_length_suffixes

    vals = _emulate_knot(lengths)
    hash_ = _densify(vals)
    return hash_


def decode_hash(hash_: str):
    res = []
    for char in hash_:
        bin_ = hex_digit2binary[char]
        res += [int(s) for s in bin_]
    return res


def find_used(key_string: str) -> list:
    """Takes a key string and returns a list of lists with ones and zeroes indicating whether a block is used."""
    res = []
    for i in range(128):
        seed = f"{key_string}-{i}"
        hash_ = knot_hash(seed)
        row = decode_hash(hash_)
        res.append(row)

    return res


def determine_groups(used):
    """Determines groups of used blocks. Returns list of sets representing the i,j-coordinates of the blocks in each
    group."""

    res = []
    missing = {(i, j) for i, row in enumerate(used) for j, val in enumerate(row) if val}
    rows = len(used)
    cols = len(used[0])
    dim = (rows, cols)

    while missing:
        group = set([])
        # Start from a random node from the ones still not assigned to a group
        wavefront = {missing.pop()}
        while wavefront:
            group = group.union(wavefront)
            next_wave = set([])
            for i, j in wavefront:
                # Check all the neighbors
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    coord = (i+di, j+dj)
                    # Ignore out of bound sites
                    if not all(0 <= x < lim for x, lim in zip(coord, dim)):
                        continue
                    # If we encounter a new missing node, accept it for the next iteration
                    if coord in missing and coord not in group:
                        next_wave.add(coord)
                    #
                #
            wavefront = next_wave
        res.append(group)
        missing -= group

    return res


def solve(data: str) -> tuple[int|str, int|str]:
    key_string = parse(data)

    used = find_used(key_string)
    star1 = sum(sum(row) for row in used)
    print(f"Solution to part 1: {star1}")

    groups = determine_groups(used)
    star2 = len(groups)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2017, 14
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
