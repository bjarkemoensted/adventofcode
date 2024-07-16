import copy


def read_input():
    with open("input13.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    """Parses input into packet pairs"""
    groups = s.split("\n\n")
    packet_pairs = [tuple(eval(substring) for substring in group.split("\n")) for group in groups]
    return packet_pairs


def in_order(a, b):
    """Determines whether packets a an b are in order."""

    a = copy.deepcopy(a)
    b = copy.deepcopy(b)

    # Inds are in order if a<b, out of order if a>b
    if all(isinstance(val, int) for val in (a, b)):
        if a != b:
            return a < b
        else:
            # None indicates an ordering can't be decided from a and b
            return None
        #

    # If a and b aren't inds, make sure they're both lists
    if isinstance(a, int):
        a = [a]
    if isinstance(b, int):
        b = [b]

    # Compare all elements of the lists
    maxlength = max([len(arr) for arr in (a, b)])
    for i in range(maxlength):
        try:
            # If either element is ordered wrt the other, we're done.
            partial_result = in_order(a[i], b[i])
            if partial_result is not None:
                return partial_result
        except IndexError:
            # If a list runs out of elements, it's 'smaller' than the other
            lengths_in_order = len(a) < len(b)
            return lengths_in_order


def find_inds_in_order(packet_pairs):
    """Determines the indices (index 1) of the packets pairs that are in order."""
    inds = []
    for i, (a, b) in enumerate(packet_pairs):
        if in_order(a, b):
            inds.append(i + 1)
        #

    return inds


def find_decoder_key(packet_pairs, divider_packets):
    """Determines the decoder key from input packet pairs and divider packets"""

    # Generate a list of all packets, including divider packets, from the packet pairs
    packets = copy.deepcopy(divider_packets)
    for a, b in copy.deepcopy(packet_pairs):
        packets.append(a)
        packets.append(b)

    # Repeatedly find the minimum element and append to ordered list (v effective O(n^2) sorting, stand back!)
    ordered = []
    while packets:
        min_ind = None
        for i, packet in enumerate(packets):
            if min_ind is None or in_order(packet, packets[min_ind]):
                min_ind = i
            #
        ordered.append(packets.pop(min_ind))

    # Obtain the decoder key by multiplying together the indices (index 1) of the divider packets
    key = 1
    for dp in divider_packets:
        dp_ind = ordered.index(dp) + 1
        key *= dp_ind

    return key


def main():
    raw = read_input()
    packet_pairs = parse(raw)
    inds_in_order = find_inds_in_order(packet_pairs)
    print(f"Sum of indices in order is {sum(inds_in_order)}.")

    divider_packets = [[[2]], [[6]]]
    decoder_key = find_decoder_key(packet_pairs, divider_packets)
    print(f"The decoder key is {decoder_key}.")


if __name__ == '__main__':
    main()
