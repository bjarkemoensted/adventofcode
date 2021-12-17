

with open("input17.txt") as f:
    # target area: x=269..292, y=-68..-44
    stuff = f.read().strip().split("target area: ")[1].split(", ")
    borders = []
    for s in stuff:
        border = tuple(int(x) for x in s.split("=")[1].split(".."))
        borders.append(border)


def within_borders(coords, borders):
    for coord, (a, b) in zip(coords, borders):
        if not (a <= coord <= b):
            return False
        #
    return True


def missed(coords, borders):
    return coords[0] > borders[0][1] or coords[1] < borders[1][0]


min_x_velocity = 