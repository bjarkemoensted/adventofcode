# Read in data
with open("input01.txt") as f:
    s = f.read()


# Star 1
d = {"(": 1, ")": -1}
print(sum(d[char] for char in s))

# Star 2
running = 0
for i, char in enumerate(s):
    running += d[char]
    if running < 0:
        print(i+1)
        break