import copy

with open("input16.txt") as f:
    hex_ = f.read()


mapping = {
    "0" : "0000",
    "1" : "0001",
    "2" : "0010",
    "3" : "0011",
    "4" : "0100",
    "5" : "0101",
    "6" : "0110",
    "7" : "0111",
    "8" : "1000",
    "9" : "1001",
    "A" : "1010",
    "B" : "1011",
    "C" : "1100",
    "D" : "1101",
    "E" : "1110",
    "F" : "1111"
}

bin_ = "".join(mapping[char] for char in hex_)


def parse_packets(s):
    i = 0
    res = []
    while i < len(s):
        print(f"Yoooo i={i}")
        d = {}
        d["packet_version"] = s[i:i+3]
        i += 3
        d["type_id"] = s[i:i+3]
        i += 3

        temp = ""
        while s[i] == "1":
            print(s[i:i+5])
            nextbits = s[i+1: i+5]
            temp += nextbits
            i += 5
            print(nextbits)
        print(s[i], s[i] == "0")

        lastbits = s[i+1: i+5]
        print(lastbits)
        temp += lastbits
        i += 5

        print(f"Before trailing: i={i}")
        if i % 4 != 0:
            trailing = 4 - i % 4
            assert all(char == "0" for char in s[i: i+trailing])
            i += trailing
        d["value"] = temp
        res.append({k: int(v, 2) for k, v in d.items()})
        print(f"End of loop: i={i}")

    return res

d = parse_packets(bin_)