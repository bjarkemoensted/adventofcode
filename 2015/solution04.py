from hashlib import md5

# Read in data
with open("input04.txt") as f:
    s = f.read()


def make_hash(x):
    str_ = x.encode('utf8')
    hash_ = md5(str_).hexdigest()
    return hash_


def find_first_hash(startstring):
    """Finds the lowest positive integer n such that the MD5 hash of puzzle input + n starts with the startstring."""
    n = 0
    while not make_hash(s+str(n)).startswith(startstring):
        n += 1
        if n % 10000 == 0:
            print(n, end="\r")
    print()
    return n


star1 = find_first_hash(5*"0")
print(f"Solution 1: {star1}.")

star2 = find_first_hash(6*"0")
print(f"Solution 1: {star2}.")
