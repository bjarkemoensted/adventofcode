# `*·. ·*.  `  · ·.* ·   *  . ` ·   · · *.· `.·  *·.*`    .•·* . ·  *  · . *··`.
# ·.+ ·*`· .    · ·  *  · ·.` +· Distress Signal `*  · ·•. .   *·  `.     ·` .+·
# *`   ..* · ·  . *`+  https://adventofcode.com/2022/day/13 ` .   *.·*· `*  .`· 
# ··  *·+ .`·   •   ·`*.   `*.·`•   ·      •·`*  ·.*    . ·  `•·  ·  `* ·  ·* .·

import copy
import json
from typing import TypeGuard

# Type alias for packet
type packettype = list[int|packettype]


def is_packet(obj, top=True) -> TypeGuard[packettype]:
    """Type guard for type narrowing packets."""
    
    if isinstance(obj, int):
        return not top  # allow int except at the top level
    elif isinstance(obj, list):
        # Recurse for lists
        return all(is_packet(elem, top=False) for elem in obj)
    else:
        return False


def parse(s: str) -> list[tuple[packettype, packettype]]:
    """Parses input into packet pairs"""

    res: list[tuple[packettype, packettype]] = []

    for group in s.split("\n\n"):
        a, b = (json.loads(substring) for substring in group.splitlines())
        assert is_packet(a)
        assert is_packet(b)
        res.append((a, b))

    return res


def in_order(a: packettype|int, b: packettype|int):
    """Determines whether packets a an b are in order."""

    a = copy.deepcopy(a)
    b = copy.deepcopy(b)

    # Inds are in order if a<b, out of order if a>b
    if isinstance(a, int) and isinstance(b, int):
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
        #
    #


def find_inds_in_order(packet_pairs: list[tuple[packettype, packettype]]) -> list[int]:
    """Determines the indices (index 1) of the packets pairs that are in order."""

    inds = []
    for i, (a, b) in enumerate(packet_pairs):
        if in_order(a, b):
            inds.append(i + 1)
        #

    return inds


def find_decoder_key(
        packet_pairs: list[tuple[packettype, packettype]],
        divider_packets: list[packettype]
        ) -> int:
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
        assert isinstance(min_ind, int)
        ordered.append(packets.pop(min_ind))

    # Obtain the decoder key by multiplying together the indices (index 1) of the divider packets
    key = 1
    for dp in divider_packets:
        dp_ind = ordered.index(dp) + 1
        key *= dp_ind

    return key


def solve(data: str) -> tuple[int|str, ...]:
    packet_pairs = parse(data)
    inds_in_order = find_inds_in_order(packet_pairs)
    star1 = sum(inds_in_order)
    print(f"Solution to part 1: {star1}")

    divider_packets: list[packettype] = [[[2]], [[6]]]
    star2 = find_decoder_key(packet_pairs, divider_packets)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 13
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
