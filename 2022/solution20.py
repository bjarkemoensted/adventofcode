from copy import deepcopy


def read_input():
    with open("input20.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = [int(line) for line in s.split("\n")]
    return res


def make_num_count_tuples(numbers):
    """Converts a list of numbers into a list of tuples of numbers and the number of times it has occurred
    in the list. Example
    [1, 1, 5] -> [(1, 1), (1, 2), (5, 1)]"""

    counts = {}
    res = []
    for number in numbers:
        n_occ = counts.get(number, 0) + 1
        res.append((number, n_occ))
        counts[number] = n_occ

    return res


def make_linked_list_ish(elems):
    """Converts element into a linked list type structure, where each element maps to a dict pointing (via 'prev' and
    'next' entries) to the previous and next elements, respectively."""

    assert len(elems) == len(set(elems))
    res = {}
    for i, number in enumerate(elems):
        d = {
            "prev": elems[(i - 1) % len(elems)],
            "next": elems[(i + 1) % len(elems)]
        }
        res[number] = d

    return res


def remove_element(linkedlist, elem):
    """Removes input element from linked list, fixing where the previous and next elements point to."""
    next_ = linkedlist[elem]["next"]
    prev = linkedlist[elem]["prev"]

    linkedlist[prev]["next"] = next_
    linkedlist[next_]["prev"] = prev
    return elem


def insert_element_right(linkedlist, number_location, tup_insert):
    """Inserts element to the right of specified element in a linked list. I.e. if inserting 5 into linked list
    1 > 2 > 3 at location 2, result will be 1 > 2 > 5 > 3."""

    next_ = linkedlist[number_location]["next"]

    # Insert element
    insert_elem = {
        "prev": number_location,
        "next": next_
    }
    linkedlist[tup_insert] = insert_elem

    # fix pointers
    linkedlist[next_]["prev"] = tup_insert
    linkedlist[number_location]["next"] = tup_insert


def to_list(linkedlist, startat=None, only_numbers=False):
    """Converts a linked list into a list. If no startat element is given, looks for a tuple where the 0'th element
    is as close as possible to 0."""

    if startat is None:
        order = sorted([a for a, b in linkedlist.keys()], key=abs)
        startat = (order[0], 1)

    res = []
    tup = startat
    # Iterate around the linked list until we're back where we started.
    while not res or tup != startat:
        elem = tup[0] if only_numbers else tup
        res.append(elem)
        tup = linkedlist[tup]["next"]

    return res


def move_element(linkedlist, elem):
    """Moves an element around, as specified in the rules, in a linked list.
    The element must be a tuple, with the 0'th element being a number denoting the number of steps the element is moved.
    For instance, in a list (1, 1) > (2, 1) > (1, 2), moving the element (2, 1) wille shift that tuple right 2 times."""

    number, n_occ = elem
    if number == 0:
        return
    key = "next" if number > 0 else "prev"

    running = elem
    next_step = linkedlist[elem][key]

    elem = remove_element(linkedlist, elem)

    # List is circular so no point iterating N times around if N divides n_elements
    n_elems = len(linkedlist)
    n_iterations = abs(number) % (n_elems - 1)
    # If we're iterating left, go one extra step (because we're inserting to the right of target element)
    if number < 0:
        n_iterations += 1

    for _ in range(n_iterations):
        running = next_step
        next_step = linkedlist[running][key]

    insert_element_right(linkedlist, running, elem)


def mix(linkedlist, move_order):
    """Moves all the elements in given linked list."""
    for number in move_order:
        move_element(linkedlist, number)

    return


def _coordinate_from_mixed_list(linkedlist):
    """Finds grove coordinate from mixed linked list."""
    mixed = to_list(linkedlist, startat=(0, 1))

    res = 0
    for ind in (1000, 2000, 3000):
        i = ind % len(mixed)
        res += mixed[i][0]

    return res


def determine_coordinate(numbers):
    """Determines grove coordinate from input numbers."""

    # Set up linked list containing the number/count tuples.
    tuples = make_num_count_tuples(numbers)
    linkedlist = make_linked_list_ish(tuples)

    # Do the mixing step
    mix(linkedlist, move_order=tuples)
    res = _coordinate_from_mixed_list(linkedlist)
    return res


def decrypt(numbers, decryption_key, n_mixings=10):
    """Decrypts grove coordinate by running the mixing steps multiple times."""
    numbers_decrypted = [number*decryption_key for number in numbers]
    tuples = make_num_count_tuples(numbers_decrypted)

    linkedlist = make_linked_list_ish(deepcopy(tuples))

    for i in range(n_mixings):
        mix(linkedlist, move_order=tuples)

    res = _coordinate_from_mixed_list(linkedlist)

    return res


def main():
    raw = read_input()
    numbers = parse(raw)

    coord = determine_coordinate(numbers)
    print(f"Coord: {coord}.")

    decryption_key = 811589153
    coord2 = decrypt(numbers, decryption_key)
    print(f"Decrypted coordinate is {coord2}.")


if __name__ == '__main__':
    main()
