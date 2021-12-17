import copy
import functools

# Read in the message in hexadecimal
with open("input16.txt") as f:
    hex_ = f.read()

# Convert into binary
mapping = {
    "0": "0000",
    "1": "0001",
    "2": "0010",
    "3": "0011",
    "4": "0100",
    "5": "0101",
    "6": "0110",
    "7": "0111",
    "8": "1000",
    "9": "1001",
    "A": "1010",
    "B": "1011",
    "C": "1100",
    "D": "1101",
    "E": "1110",
    "F": "1111"
}

bin_ = "".join(mapping[char] for char in hex_)


class Parser:
    """Parser base class"""

    def __init__(self, s, verbose=False, level=0, remove_trailing_zeroes=True):
        self.s = copy.copy(s)  # The string to parse
        self.ind = 0  # Points to the current character being considered
        self.parsed = {}  # Parsed results
        self.verbose = verbose
        self.level = level  # How many levels into subpackets are we?
        self.remove_trailing_zeroes = remove_trailing_zeroes  # Whether to trim trailing zeroes from packet

    def vprint(self, *args, **kwargs):
        if self.verbose:
            spacechar = '-'
            space = 2 * self.level * spacechar
            print(space, *args, **kwargs)

    def _scan_ahead(self, nchars):
        """Scans n characters forward in the message and returns the resulting substring"""
        a = self.ind
        self.ind += nchars
        b = self.ind
        substring = self.s[a:b]
        # self.vprint(f"Scanned {nchars} chars: {substring}.")
        return substring

    def scan_packet_version(self):
        """Scans the packet version (initial first 3 chars)"""
        s = self._scan_ahead(3)
        val = int(s, 2)
        self.vprint(f"Got packet version {s} ({val}).")
        self.parsed["packet_version"] = val

    def scan_type_id(self):
        """Scans type ID (chars 3:6)"""
        s = self._scan_ahead(3)
        val = int(s, 2)
        self.vprint(f"Got type ID: {s} ({val}).")
        self.parsed["type_id"] = val

    def scan_trailing_zeroes(self):
        """Fixes tralining zeroes"""
        if self.ind % 4 != 0:
            trailing = 4 - self.ind % 4
            zeroes = self._scan_ahead(trailing)
            assert all(zero == "0" for zero in zeroes)
            self.vprint(f"Got {len(zeroes)} trailing zeroes.")
        else:
            self.vprint("Got no trailing zeroes")

    def parse(self):
        raise NotImplementedError("Override this!")


class LiteralValueParser(Parser):
    def scan_number(self):
        """Parses a number by reading in 'chunks' of 5 bits. First is a continuation bit - keep scanning
        if 1. Reamining 4 bits represent part of the number. Keep going until continuation bit is 0."""
        bits = ""
        while self.s[self.ind] == "1":
            bits += self._scan_ahead(5)[1:]
        bits += self._scan_ahead(5)[1:]
        val = int(bits, 2)
        self.parsed["value"] = val
        self.vprint(f"Got number: {bits} ({val}).")

    def parse(self):
        """Parse a literal value packet (type 4)"""
        self.scan_packet_version()
        self.scan_type_id()
        self.scan_number()
        if self.remove_trailing_zeroes:
            self.scan_trailing_zeroes()
        return self.parsed


def next_packet_is_literal_value(s):
    return s[3:6] == "100"


class OperatorParser(Parser):
    def scan_length_type(self):
        """Scan the packet length type (7th character)"""
        char = self._scan_ahead(1)
        val = int(char, 2)
        self.parsed["length_type"] = val
        self.vprint(f"Got length type {char} ({val}).")

    def scan_subpacket_length(self):
        """Determines the subpacket length."""
        nchars = 15 if self.parsed["length_type"] == 0 else 11
        len_ = self._scan_ahead(nchars)
        val = int(len_, 2)
        self.parsed["subpacket_length"] = val
        self.vprint(f"Got subpacket length: {len_} ({val}).")

    def scan_next_subpacket(self):
        """Scans the next subpacket."""
        # Until consider the remainder of the string being parsed
        substring = copy.copy(self.s[self.ind:])
        # Create a parser for the substring
        subparser = None
        kwargs = dict(s=substring, verbose=self.verbose, level=self.level + 1, remove_trailing_zeroes=False)
        if next_packet_is_literal_value(substring):
            subparser = LiteralValueParser(**kwargs)
        else:
            subparser = OperatorParser(**kwargs)

        # Parse the subpacket
        res = subparser.parse()
        # Increment the index with the number of characters of the subpacket
        subpacket_len = subparser.ind
        self.ind += subpacket_len
        return res

    def scan_subpackets_bits(self, nchars):
        """Scan subpackets until nchars characters have been used."""
        ind_start = self.ind
        subpackets = []
        while self.ind - ind_start < nchars:
            subpacket = self.scan_next_subpacket()
            subpackets.append(subpacket)
        assert self.ind - ind_start == nchars
        self.parsed["subpackets"] = subpackets

    def scan_subpackets_n(self, n):
        """Scans n subpackets"""
        subpackets = []
        while len(subpackets) < n:
            subpacket = self.scan_next_subpacket()
            subpackets.append(subpacket)
        self.parsed["subpackets"] = subpackets

    def parse(self):
        # Do simple parsing stuff
        self.scan_packet_version()
        self.scan_type_id()
        self.scan_length_type()
        self.scan_subpacket_length()

        subpacket_len = self.parsed["subpacket_length"]
        # If length type 0, subpacket length tells us how many bits we need to parse for the subpackets
        if self.parsed["length_type"] == 0:
            self.scan_subpackets_bits(subpacket_len)
        # Otherwise, it tells us the number of subpackets
        else:
            self.scan_subpackets_n(subpacket_len)

        return self.parsed


def recursive_iterate(d, value_field="packet_version", level_field="subpackets"):
    """Recursively iterates through nested dictionaries.
    level_field is the key pointing to the next level down.
    value_field is the value yielded from the dicts."""
    yield d[value_field]
    for packet in d.get(level_field, []):
        for item in recursive_iterate(packet, value_field=value_field, level_field=level_field):
            yield item


# Parse the message
parser = OperatorParser(s=bin_, verbose=False)
parsed = parser.parse()

# Sum the packet version numbers in all the (sub)packets.
star1 = sum(recursive_iterate(parsed))
print(f"Solution to star 1: {star1}.")

# Map packet types to the operation which must be done on its subpackets
funs = {
    0: sum,
    1: lambda arr: functools.reduce(lambda a, b: a * b, arr),
    2: min,
    3: max,
    4: lambda x: x,
    5: lambda arr: int(arr[0] > arr[1]),
    6: lambda arr: int(arr[0] < arr[1]),
    7: lambda arr: int(arr[0] == arr[1])
}


# Recursively aggregate the results as per the functions above
def crunch(d):
    type_ = d["type_id"]
    if type_ == 4:
        return d["value"]
    else:
        fun = funs[type_]
        subvals = [crunch(packet) for packet in d["subpackets"]]
        return fun(subvals)
    #


# Phew
star2 = crunch(parsed)
print(f"Solution to star 2: {star2}.")