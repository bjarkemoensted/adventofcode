# Read in data
with open("input02.txt") as f:
    raw = f.read()


def parse(s):
    res = [tuple(int(elem) for elem in line.split("x")) for line in s.split("\n")]
    return res


def compute_wrapping_paper_area(dimensions):
    a, b, c = dimensions
    sides = [a*b, b*c, a*c]
    area = 2*sum(sides)
    slack = min(sides)
    return area + slack


def compute_ribbon_length(dimensions):
    a, b, c = dimensions
    volume = a*b*c
    shortest_edges = sorted(dimensions)[:2]
    circumference = 2*sum(shortest_edges)
    return circumference + volume


data = parse(raw)
areas = [compute_wrapping_paper_area(dimension) for dimension in data]
print(f"Need at total of {sum(areas)} square feet of wrapping paper.")

ribbon_length = sum(compute_ribbon_length(dimension) for dimension in data)
print(f"Need {ribbon_length} feet of ribbon.")

