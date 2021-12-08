digits_list = []
displays_list = []

with open("input08.txt") as f:
    for line in f.readlines():
        a, b = [elem.split() for elem in line.split(" | ")]
        digits_list.append(a)
        displays_list.append(b)
    #

digit2segments={
    0: 'abcefg',
    1: 'cf',
    2: 'acdeg',
    3: 'acdfg',
    4: 'bcdf',
    5: 'abdfg',
    6: 'abdefg',
    7: 'acf',
    8: 'abcdefg',
    9: 'abcdfg'
}

digit2n_segments = {k: len(v) for k, v in digit2segments.items()}

target_digits = [1, 4, 7, 8]
target_lengths = {len(digit2segments[k]) for k in target_digits}

n_target = sum(len(s) in target_lengths for s in sum(displays_list, []))
print(f"Solution to star 1: {n_target}.")


def crack(digits, displays):
    all_numbers = list(range(10))


digits2overlap = {}
for k1, v1 in digit2segments.items():
    for k2, v2 in digit2segments.items():
        if k2 <= k1:
            continue
        digits2overlap[(k1, k2)] = len(set(v2).intersection(set(v1)))

numbers = []
for digits, displays in zip(digits_list, displays_list):
    number = crack(digits, displays)
    numbers.append(number)
    break

print(numbers)