from ast import literal_eval as LE
import math

with open("input18.txt") as f:
    raw = f.read()


def parse(s):
    res = [LE(line.strip()) for line in s.split("\n")]
    return res


nums = parse(raw)


def add(a, b):
    res = [a, b]
    return res


def access(arr, inds):
    ptr = arr
    for i in inds:
        ptr = ptr[i]
    return ptr


def set_(arr, inds, val):
    ptr = arr
    for i in inds[:-1]:
        ptr = ptr[i]
    ptr[inds[-1]] = val


def determine_starting_ind(arr):
    res = [0]
    while isinstance(access(arr, res), list):
        res.append(0)
    return res


def iterate_inds(arr, start=None, right=True):
    inds = start
    if inds is None:
        inds = determine_starting_ind(arr)
    inds = [ind for ind in inds]
    step = 1 if right else -1
    offset = 0 if right else 1

    while inds:
        inds[-1] += step
        val = None
        try:
            if inds[-1] < 0:
                raise IndexError
            val = access(arr, inds)
        except IndexError:
            inds.pop()
        if isinstance(val, list):
            inds.append(offset)

        yield inds


def find_first_number_ind(arr, start, right=True):
    start = [val for val in start]
    for inds in iterate_inds(arr, start, right):
        val = access(arr, inds)
        if isinstance(val, int):
            return inds
        #
    return None


def explode(arr, inds):
    x = access(arr, inds)
    assert isinstance(x, list)
    startinds = inds

    for num, right in zip(x, (False, True)):
        nextinds = find_first_number_ind(arr, startinds, right=right)
        if nextinds is None:
            continue
        current = access(arr, nextinds)
        new_val = num + current
        set_(arr, nextinds, new_val)
    set_(arr, startinds, 0)


def split(arr, inds):
    val = access(arr, inds)
    assert isinstance(val, int)
    x = val/2
    a = math.floor(x)
    b = math.ceil(x)
    new_pair = [a, b]
    set_(arr, inds, new_pair)


def try_explode(arr):
    for inds in iterate_inds(arr):
        stuff = access(arr, inds)
        if len(inds) >= 4 and isinstance(stuff, list) and all(isinstance(elem, int) for elem in stuff):
            print(f"Exploding {arr} at location {inds} (value {access(arr, inds)})")
            explode(arr, inds)
            return True
        #
    return False


def try_split(arr):
    for inds in iterate_inds(arr):
        val = access(arr, inds)
        if isinstance(val, int) and val >= 10:
            print(f"Splitting {arr} at location {inds} (value {access(arr, inds)})")
            split(arr, inds)
            return True
    return False


def reduce(arr):
    done = False
    while not done:
        if try_explode(arr):
            continue
        elif try_split(arr):
            continue
        else:
            done = True
        #
    return


def compute_final_sum(numbers):
    running = numbers[0]
    reduce(running)
    for number in numbers[1:]:
        running = add(running, number)
        reduce(running)
    return running


s = """[[[0,[4,5]],[0,0]],[[[4,5],[2,6]],[9,5]]]
[7,[[[3,7],[4,3]],[[6,3],[8,8]]]]
[[2,[[0,8],[3,4]]],[[[6,7],1],[7,[1,6]]]]
[[[[2,4],7],[6,[0,5]]],[[[6,8],[2,8]],[[2,1],[4,5]]]]
[7,[5,[[3,8],[1,4]]]]
[[2,[2,2]],[8,[8,1]]]
[2,9]
[1,[[[9,3],9],[[9,0],[0,7]]]]
[[[5,[7,4]],7],1]
[[[[4,2],2],6],[8,7]]"""

nums = parse(s)[:2]

meh = compute_final_sum(nums)
