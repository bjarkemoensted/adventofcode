from copy import deepcopy


def parse(s):
    res = s
    return res


_trap_patterns = {
    "^^.",  # only left and center are traps
    ".^^",  # Only right and center are traps
    "^..",  # Only left tile is a trap
    "..^"   # Only right tile is a trap
}


def next_tile_type(previous_tiles: str) -> str:
    """Given a string of 3 tiles in the preceding row, returns a character representing the next tile type."""
    if len(previous_tiles) != 3:
        raise ValueError
    is_trap = previous_tiles in _trap_patterns
    res = "^" if is_trap else "."
    return res


def determine_next_tile_row(row: str) -> str:
    """Takes a string representing a row of tiles. Returns a string representing the next row."""

    # No trap tiles outside the corridor, so we pad the row with 'imaginary tiles' which are safe
    pad = "."
    row_with_padding = pad+row+pad

    # Determine all tiles in the next row and return
    res = "".join([next_tile_type(row_with_padding[i-1:i+2]) for i in range(1, len(row_with_padding) - 1)])
    if len(res) != len(row):
        raise ValueError

    return res


def compute_number_of_safe_tiles(initial_row: str, n_rows_total: int):
    """Computes the number of safe tiles in a room given an initial row, and number of rows of tiles in the room.
    Iteratively grows new rows and notes the number of safe tiles.
    If a tile has been encountered previously, the resulting pattern recurrence is used to compute the result
    more efficiently for very large rooms."""

    row2rownumber = dict()  # Maps each row to the index where it was first encountered
    n_safe_by_row = []  # Number of safe tiles in each row

    # Keep generating new rows until the required number of rows is reached, or a recurring pattern is identified
    ind = 0
    row = initial_row
    while ind < n_rows_total and row not in row2rownumber:
        # Update the data on row indices and number of safe tiles
        row2rownumber[row] = ind
        n_safe = row.count(".")
        n_safe_by_row.append(n_safe)

        # Move on to the next row
        row = determine_next_tile_row(row)
        ind += 1

    # Compute the total number of safe tiles  encountered thus far
    res = sum(n_safe_by_row[:ind])

    # If a recurring pattern was spotted, add the number of safe tiles which doesn't need to be explicitly generated
    recurrence_index = row2rownumber.get(row)
    if recurrence_index is not None:
        # Identify the rows which will repeat
        rows_in_repeating_part = n_safe_by_row[recurrence_index: ind]
        n_rows_missing = n_rows_total - ind

        # Add to the result the number of safe tiles in the entire pattern for each number of remaining pattern repeats
        n_repeats = n_rows_missing // len(rows_in_repeating_part)
        res += n_repeats*sum(rows_in_repeating_part)

        # The pattern might end with only a partial repetition of the pattern. Add the safe tiles in any partial reps
        n_remainder = n_rows_missing % len(rows_in_repeating_part)
        res += sum(rows_in_repeating_part[:n_remainder])

    return res


def solve(data: str):
    example_rows = {5: 3, 10: 10}
    initial_row = parse(data)
    n_rows = example_rows.get(len(data), 40)
    star1 = compute_number_of_safe_tiles(initial_row, n_rows)

    print(f"The room contains {star1} safe tiles in a room with {n_rows} rows.")

    n_rows2 = 400000
    star2 = compute_number_of_safe_tiles(initial_row, n_rows2)
    print(f"A similar room with {n_rows2} has {star2} safe tiles.")

    return star1, star2


def main():
    year, day = 2016, 18
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve, extra_kwargs_parser="ignore")
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
