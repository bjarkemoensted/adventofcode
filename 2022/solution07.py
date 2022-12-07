def read_input():
    with open("input07.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def mkdir(d, name_):
    """Adds a directory to directory. Also adds a link to parent, keyed by '..'"""
    d[name_] = {}
    d[name_][".."] = d


def add_file(d, name_, size):
    """Adds a file (filename, fizesize) to directory"""
    d[name_] = (name_, size)


def isint(s):
    """Checks if string can be cast as int"""
    try:
        _ = int(s)
        return True
    except ValueError:
        return False


def parse(s):
    """Parses input data into a recursive file structure dicts"""
    d = {}
    pwd = d
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
                    pwd = pwd[b]
        else:
            a, b = line.strip().split(" ")
            # Create a subdirectory if a dir is listed
            if a == "dir":
                mkdir(pwd, b)
            # Create a file if a file is listed
            elif isint(a):
                add_file(pwd, name_=b, size=int(a))
            else:
                raise ValueError

    return d


def compute_directory_size(d):
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


def traverse(d):
    """Recursively traverses directory.
    generates all (directory_name, directory_dict) tuples."""

    for k, v in d.items():
        if k == "..":
            continue
        if isinstance(v, dict):
            yield k, v
            yield from traverse(v)


def main():
    # Parse input into root directory
    raw = read_input()
    d = parse(raw)

    # Make list of all directory sizes
    sizes = []
    for dirname, dir_ in traverse(d):
        size = compute_directory_size(dir_)
        sizes.append(size)

    # Sum the ones less than the threshold given in the problem
    small_dirs = [size for size in sizes if size <= 100000]
    print(f"Small directories sum to {sum(small_dirs)}.")

    # Figure out the minimum amount of disk space we need to free up
    total_space_used = compute_directory_size(d)
    disk_capacity = 70000000
    space_needed = 30000000
    target_space = disk_capacity - space_needed
    need_to_delete = total_space_used - target_space

    # Find the smallest directory size which is still large enough to free up the required space
    candidates = [size for size in sizes if size >= need_to_delete]
    smallest_candidate = min(candidates)
    print(f"Smallest directory which frees up enough space takes up: {smallest_candidate}.")


if __name__ == '__main__':
    main()
