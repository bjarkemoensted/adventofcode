# *. · `+· *.   `* ·  `.  +   `      ·.`+   *.`·      `    ·. `+·  *     · .. •`
# · .*`    .   · `. ·      * `+. . Lens Library  `.*·.·* `    + `·.  ·   *`   .·
# `·`+ ·.* `   .. ·  · https://adventofcode.com/2023/day/15  `· .* `·*.  `.·`+  
#  `·. *     ·   .` * ·  .` ·  ·` *+. `·    `·   .  *`.·`+     ·    *` ·.  * `·.


def parse(s: str) -> list[str]:
    res = s.split(",")
    return res


def hash_(s):
    """Computes the Holiday ASCII String Helper algorithm output for a string"""
    res = 0
    for char in s:
        res = ((res + ord(char))*17) % 256
    return res


def get_box_ind(instruction):
    s = None
    if "=" in instruction:
        s = instruction.split("=")[0]
    elif instruction.endswith("-"):
        s = instruction[:-1]
    return hash_(s)


def label_type(label):
    """Returns the label type of a label, e.g. 'cm 1' -> 'cm'."""
    return label.split(" ")[0]


def put(box, lens_label):
    """Puts a lens (defined by its label) into the given box.
    Replaces the lens with the same type if one is present. Otherwise, adds lens to the end."""
    type_ = label_type(lens_label)
    for i, elem in enumerate(box):
        if label_type(elem) == type_:
            box[i] = lens_label
            return
        #
    box.append(lens_label)


def remove(box, type_):
    """Remove from the specified box the lens of the input type if one is present."""
    for i, elem in enumerate(box):
        if label_type(elem) == type_:
            box.pop(i)
            return


def display(boxes):
    """Helper method for displaying the state of the boxes."""
    lines = []
    for i, box in enumerate(boxes):
        if not box:
            continue
        labels = " ".join([f"[{label}]" for label in box])
        line = f"Box {i}: {labels}"
        lines.append(line)
    res = "\n".join(lines)
    print(res)


def install_lenses(instructions):
    """Puts lenses into the boxes as specified in the instructions"""
    boxes = [[] for _ in range(256)]
    for ins in instructions:
        i = get_box_ind(ins)
        if "=" in ins:
            lens_label = ins.replace("=", " ")
            put(boxes[i], lens_label)
        elif ins.endswith("-"):
            type_ = ins[:-1]
            remove(boxes[i], type_)
        else:
            raise ValueError
        #display(boxes)
        #print()
    return boxes


def compute_focusing_power(boxes):
    res = 0
    for i, box in enumerate(boxes):
        for j, lens_label in enumerate(box):
            focal_length = int(lens_label.split()[-1])
            res += (i + 1)*(j + 1)*focal_length
        #

    return res


def solve(data: str) -> tuple[int|str, ...]:
    instructions = parse(data)

    star1 = sum(map(hash_, instructions))
    print(f"Solution to part 1: {star1}")

    boxes = install_lenses(instructions)
    star2 = compute_focusing_power(boxes)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 15
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
