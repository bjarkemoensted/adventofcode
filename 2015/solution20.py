# Read in data
with open("input20.txt") as f:
    puzzle_input = int(f.read())


def find_divisors(n, max_prod=None):
    """Returns list of divisors given a number.
    max_prod specifies the maximum allowed division, e.g. if a divides b, a will not
    be included in results if b // a > max_prod."""

    if not isinstance(n, int):
        raise TypeError
    
    if max_prod is None:
        max_prod = float('inf')

    large_divisors = []
    small_divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            if n // i <= max_prod:
                small_divisors.append(i)
            if i*i != n:
                large_div = n // i
                if n // large_div <= max_prod:
                    large_divisors.append(large_div)
        i += 1
    
    divisors = small_divisors + large_divisors[::-1]
    return divisors


def compute_n_presents(house, multiply_by=10, max_prod=None):
    divisors = find_divisors(house, max_prod=max_prod)
    presents = multiply_by*sum(divisors)
    return presents

    
def find_first_house_receiving_n_presents(n_presents, multiply_by=10, max_prod=None):
    house = 1
    while compute_n_presents(house, multiply_by=multiply_by, max_prod=max_prod) < n_presents:
        house += 1
        if house % 10000 == 0:
            print(f"Reached house {house}.", end="\r")
        #
    print()

    return house


n_pres = puzzle_input
house1 = find_first_house_receiving_n_presents(n_pres)
print(f"First house to receive at least {n_pres} presents is house number {house1}.")  # 776160
print()

house2 = find_first_house_receiving_n_presents(n_presents=n_pres, multiply_by=11, max_prod=50)
print(f"First house getting at least {n_pres} with the updated rules is house number {house2}.")  # 786240
