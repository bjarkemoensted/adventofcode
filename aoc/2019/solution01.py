import math

with open("input01.txt") as f:
    numbers = [int(line.strip()) for line in f]


def compute_fuel_requirement(number):
    res = math.floor(number/3) - 2
    return res


fuel_reqs = [compute_fuel_requirement(number) for number in numbers]
star1 = sum(fuel_reqs)
print(star1)


def add_fuel(amount):
    inc = compute_fuel_requirement(amount)
    if inc <= 0:
        res = amount
    else:
        res = amount + add_fuel(inc)
    return res


star2 = sum([add_fuel(number) for number in fuel_reqs])
print(star2)
