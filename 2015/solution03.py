# Read in data
with open("input03.txt") as f:
    raw = f.read()


def parse(s):
    res = list(s)
    return res


def update_coords(coords, instruction):
    x, y = coords
    if instruction == "^":
        y += 1
    elif instruction == "v":
        y -= 1
    elif instruction == ">":
        x += 1
    elif instruction == "<":
        x -= 1
    else:
        raise ValueError

    res = (x, y)
    return res


data = parse(raw)
startat = (0, 0)
current_pos = startat
pos2n_presents = {current_pos: 1}

for instruction in data:
    current_pos = update_coords(current_pos, instruction)
    pos2n_presents[current_pos] = pos2n_presents.get(current_pos, 0) + 1

print(f"Presents delivered to {len(pos2n_presents)} houses.")

current_positions = [startat, startat]
present_data = [{startat: 1} for pos in current_positions]
for i, instruction in enumerate(data):
    ind = i % 2

    current_positions[ind] = update_coords(current_positions[ind], instruction)
    d = present_data[ind]
    d[current_positions[ind]] = d.get(current_positions[ind], 0) + 1


santa, robo_santa = present_data
distinct_houses = set(santa.keys()).union(robo_santa.keys())
print(f"Santa and robo-santa delivered presents to {len(distinct_houses)} between them.")
