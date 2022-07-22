from hashlib import md5

# Read in data
with open("input05.txt") as f:
    raw = f.read()


def parse(stuff):
    res = [line for line in stuff.split("\n")]
    return res


def string_is_nice(s):
    """Determines if a string is 'nice', as specified by Santa."""
    vowels = 'aeiou'
    forbidden = ['ab', 'cd', 'pq', 'xy']

    # Must have 3 vowels
    if sum(char in vowels for char in s) < 3:
        return False
    # Must have repeating letters
    if not any(s[i] == s[i+1] for i in range(len(s)-1)):
        return False
    # Must not contain naughty substrings
    if any(nope in s for nope in forbidden):
        return False

    return True


strings = parse(raw)

n_nice = sum(string_is_nice(s) for s in strings)
print(f"There are {n_nice} nice strings.")


def string_is_nice2(s):
    if not any(s[i:i+2] in s[i+2:] for i in range(len(s) - 2)):
        return False
    if not any(s[i] == s[i+2] for i in range(len(s) - 2)):
        return False
    return True


n_nice2 = sum(string_is_nice2(s) for s in strings)
print(f"There are now {n_nice2} nice strings.")
