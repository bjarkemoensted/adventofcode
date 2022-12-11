import copy
import math


def read_input():
    with open("input11.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    """Parses puzzle input into dictionary a dictionary where keys are the monkey numbers
    and values are dicts representing the states of each monkey."""

    monkeys = {}
    chunks = s.split("\n\n")
    for chunk in chunks:
        monkey = {}
        lines = chunk.split("\n")
        monkey_number = int(lines[0].split("Monkey ")[-1][:-1])

        # List of items currently held by the monkey
        items = [int(stuff) for stuff in lines[1].split("Starting items: ")[-1].split(", ")]
        monkey["items"] = items

        # Function to update the 'worry level' when the monkey examines objects
        op = lines[2].split("Operation: new = ")[-1]
        fun = eval('lambda old: '+op)
        monkey["operation"] = fun

        # Get the divisor the monkey uses to determine to whom an items should be thrown
        div = int(lines[3].split("Test: divisible by ")[-1])
        monkey["divisor"] = div

        # Function to determine recipient monkey
        truepass = int(lines[4].split("If true: throw to monkey ")[-1])
        falsepass = int(lines[5].split("If false: throw to monkey ")[-1])
        # Need closure to get the function to work correctly, or they'll all be identical
        target = (lambda a, b, d: lambda x: a if x % d == 0 else b)(truepass, falsepass, div)
        monkey["determine_throw"] = target

        # Counter for the number of items inspected
        monkey["n_items_inspected"] = 0

        # Store the monkey in the monkey dict
        monkeys[monkey_number] = monkey

    return monkeys


def iterate(monkeys, n_rounds=1):
    """Iterates n rounds in the Keep Away game."""
    monkeys = copy.deepcopy(monkeys)
    order = sorted(monkeys.keys())

    for _ in range(n_rounds):
        for n in order:
            monkey = monkeys[n]
            for item in monkeys[n]["items"]:
                # Use the monkeys update function to determine my new worry level over the item
                new_worry = monkey["operation"](item)
                monkeys[n]["n_items_inspected"] += 1
                # Calm down a bit after inspection
                new_worry = new_worry // 3

                # Throw the item to the next monkey
                target_monkey = monkey["determine_throw"](new_worry)
                monkeys[target_monkey]["items"].append(new_worry)

            # Empty the list of items held
            monkeys[n]["items"] = []
        #

    return monkeys


def compute_monkey_business(monkeys):
    """Compute Monkey Business Score (product of two largest n items inspected)."""
    total_inspections = [d["n_items_inspected"] for d in monkeys.values()]
    a, b = sorted(total_inspections, reverse=True)[:2]
    return a*b


def _compute_least_common_multiple(numbers):
    """Finds the smallest number which all input numbers divide."""
    res = numbers[0]
    for num in numbers[1:]:
        res = res*num // math.gcd(res, num)

    return res


def iterate2(monkeys, n_rounds=1):
    """Same as the other iteration function, but we don't calm down after item inspection anymore."""
    monkeys = copy.deepcopy(monkeys)
    order = sorted(monkeys.keys())

    divs = [d["divisor"] for d in monkeys.values()]
    # Determine least common multiple of all the monkeys' divisors
    lcm = _compute_least_common_multiple(divs)

    for _ in range(n_rounds):
        for n in order:
            monkey = monkeys[n]
            for item in monkeys[n]["items"]:
                new_worry = monkey["operation"](item)
                monkeys[n]["n_items_inspected"] += 1

                # all divisors divide lcm, so any factor of that is irrelevant to monkey logic
                if new_worry // lcm >= 2:
                    new_worry = lcm + new_worry % lcm

                target_monkey = monkey["determine_throw"](new_worry)
                monkeys[target_monkey]["items"].append(new_worry)

            monkeys[n]["items"] = []
        #

    return monkeys



def main():
    raw = read_input()
    monkeydict = parse(raw)

    n_rounds = 20
    endstate = iterate(monkeydict, n_rounds=n_rounds)
    mb = compute_monkey_business(endstate)
    print(f"Monkey business value after {n_rounds} is {mb}.")

    n_rounds2 = 10000
    endstate2 = iterate2(monkeydict, n_rounds=n_rounds2)
    mb2 = compute_monkey_business(endstate2)
    print(f"Monkey business value after {n_rounds2} is {mb2}.")


if __name__ == '__main__':
    main()
