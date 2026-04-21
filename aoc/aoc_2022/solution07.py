# ·   `·   • · `.·     *.   ··.`    *` ·.+`·.  ·  ·  `   *· ·   ·`+   . `· *`··.
# *`·.· . *`     `·  . `·    No Space Left On Device  ·   ` * · .+·     ·.` ·.*`
# ·.+·`. ·`  *··`. ·   https://adventofcode.com/2022/day/7 . `.·   *· .·*  ·*``·
# .· ` ··*  ·`.     ·•`.`· · *      `.·  ·  . `·   . · . ·+`  ·.* .· ·  .`*`.·. 

from typing import Iterator

type filetype = tuple[str, int]
type dirtype = dict[str, filetype|dirtype]


def mkdir(d: dirtype, name_: str) -> None:
    """Adds a directory to directory. Also adds a link to parent, keyed by '..'"""
    new_dir: dirtype = {"..": d}
    d[name_] = new_dir
    

def add_file(d: dirtype, name_: str, size: int) -> None:
    """Adds a file (filename, fizesize) to directory"""
    d[name_] = (name_, size)


def parse(s: str) -> dirtype:
    """Parses input data into a recursive file structure dicts"""
    d: dirtype = {}
    pwd: filetype|dirtype = d

    for line in s.split("\n"):
        # Line represents a terminal command
        if line.startswith("$ "):
            instructions = line[2:].strip().split(" ")
            if len(instructions) == 1:
                # If command is ls, we don't need to do anything
                cmd = instructions[0]
                assert cmd == "ls"
            else:
                # Otherwise, command should be cd. Change current directory correspondingly
                a, b = instructions
                assert a == "cd"
                if b == "/":
                    pwd = d
                else:
                    assert isinstance(pwd, dict)
                    pwd = pwd[b]
                #
            #
        else:
            a, b = line.strip().split(" ")
            # Create a subdirectory if a dir is listed
            assert isinstance(pwd, dict)
            if a == "dir":
                mkdir(pwd, b)
            # Create a file if a file is listed
            else:
                add_file(pwd, name_=b, size=int(a))
            #
        #

    return d


def compute_directory_size(d: dirtype) -> int:
    """Computes size of the specified directory. Recursively adds file sizes from subfolders."""
    size = 0
    for k, v in d.items():
        if k == "..":
            continue
        if isinstance(v, tuple):
            size += v[1]
        elif isinstance(v, dict):
            size += compute_directory_size(v)
        #
    return size


def traverse(d: dirtype) -> Iterator[tuple[str, dirtype]]:
    """Recursively traverses directory.
    generates all (directory_name, directory_dict) tuples."""

    for k, v in d.items():
        if k == "..":
            continue
        if isinstance(v, dict):
            yield k, v
            yield from traverse(v)
        #
    #


def solve(data: str) -> tuple[int|str, ...]:
    d = parse(data)

    # Make list of all directory sizes
    sizes = []
    for dirname, dir_ in traverse(d):
        size = compute_directory_size(dir_)
        sizes.append(size)

    # Sum the ones less than the threshold given in the problem
    small_dirs = [size for size in sizes if size <= 100_000]
    star1 = sum(small_dirs)
    print(f"Solution to part 1: {star1}")

    # Figure out the minimum amount of disk space we need to free up
    total_space_used = compute_directory_size(d)
    disk_capacity = 70_000_000
    space_needed = 30_000_000
    target_space = disk_capacity - space_needed
    need_to_delete = total_space_used - target_space

    # Find the smallest directory size which is still large enough to free up the required space
    candidates = [size for size in sizes if size >= need_to_delete]
    star2 = min(candidates)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 7
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
