import json
# Read in data
with open("input12.txt") as f:
    puzzle_input = f.read()


def parse(s):
    return json.loads(s)


def recursive_iterate(obj):
    if isinstance(obj, int):
        yield obj
    elif isinstance(obj, list):
        for elem in recursive_iterate(obj):
            yield elem
        #
    elif isinstance(obj, dict):
        for elem in recursive_iterate(list(obj.values())):
            yield elem


def grab_numbers(obj, res=None, ignore_red=False):
    if res is None:
        res = []

    if isinstance(obj, int):
        res.append(obj)
    elif isinstance(obj, dict):
        if not (ignore_red and 'red' in obj.values()):
            for k, v in obj.items():
                grab_numbers(v, res=res, ignore_red=ignore_red)
    elif isinstance(obj, list):
        for elem in obj:
            grab_numbers(elem, res=res, ignore_red=ignore_red)
    return res


data = parse(puzzle_input)
numbers = grab_numbers(data)

print(f"Numbers in data sum to {sum(numbers)}.")

new_numbers = grab_numbers(data, ignore_red=True)
print(f"Non-red numbers in data sum to {sum(new_numbers)}.")
