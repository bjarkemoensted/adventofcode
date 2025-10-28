# `·· +·  `   `   ·.`•·* `  .· `·  +··. `* ·     ·.`  ·`·      ·   .· + ··` ` *·
# ·  .` ·`·* .  ·· *.*  ·  ` . · Disk Fragmenter  · ·`    *·  +.` ·· `·   ·.+·`•
#  `· ·•.  ·` ·* . `·  https://adventofcode.com/2024/day/9 .  ·` ·.* ·  +*.`·•·.
# ·+`·.` · . ·*`+  ·* · · * `  • .`· * `.·  *  ·   ·`.* · ·`   ··`  *·. ` *· . ·


def parse(s: str):
    res = [int(elem) for elem in s]
    return res


def files_as_blocks(disk_map: list) -> list:
    """Converts the dense format into a list of blocks, with elements being None for empty space, and
    an integer fileID for files"""

    running = 0
    res = []
    
    for i, datum in enumerate(disk_map):
        if i % 2 == 0:
            res += datum*[running]
            running += 1
        else:
            res += datum*[None]
        #
    
    return res


def iterate_sectors(blocks: list):
    """Iterates over sectors with identical values on a disk.
    Yields tuples like (ind_start, n_blocks, value), with value being fileID for files,
    and None for space, so e.g. (42, 3, None) indicates 3 blocks of free space at inds (42, 43, 44)."""
    
    i = 0
    while 0 <= i < len(blocks):
        val = blocks[i]
        j = i + 1
        while j < len(blocks) and blocks[j] == val:
            j += 1
        
        n_blocks = j - i
        yield i, n_blocks, val
        i += n_blocks
    #


def compress(blocks: list, allow_fragmentation=True) -> list:
    """Compresses the input blocks. If allow_defragmentation, considers each block in a file
    separately. Otherwise, only allows moving complete files into empty areas large enough
    to accomodate them."""

    # Identify the indices where a gap or file starts, and the number of blocks before it ends
    blocks = [val for val in blocks]
    gaps = []
    files = []
    
    for sector in iterate_sectors(blocks=blocks):
        i, n_blocks, value = sector
        if value is None:
            gaps.append((i, n_blocks))
        else:
            files.append((i, n_blocks))
        #
    #
    
    def iterfiles():
        """Helper method for iterating over the files to be moved (a single block at a time if defrag is ok)"""
        for ind, n_blocks in files[::-1]:
            if allow_fragmentation:
                yield from ((ind+i, 1) for i in range(n_blocks-1, -1, -1))
            else:
                yield ind, n_blocks
            #
        #
    
    def _find_space(n_blocks):
        """Identifies the first gap with sufficient space to accomodate n blocks"""
        for i, (_, size) in enumerate(gaps):
            if size >= n_blocks:
                return i
            #
        #
    
    # Starting from the last file, move into the first available space
    for file in iterfiles():
        # Look for a gap large enough for the data
        fi, n_blocks = file
        space_ind = _find_space(n_blocks=n_blocks)
        
        # If no space, proceed to the next file
        if space_ind is None:
            continue
        
        # If the space is located after the file, proceed to the next file
        gi, space = gaps[space_ind]
        if gi >= fi:
            continue
        
        # Move the block(s) into the available space
        for offset in range(n_blocks):
            a = gi + offset
            b = fi + offset
            blocks[a], blocks[b] = blocks[b], blocks[a]
        
        # Update the list of gaps
        space = space - n_blocks
        if space == 0:
            # If the gap is used completely, remove it from the list
            del gaps[space_ind]
        else:
            # Otherwise, update the block where the space starts, and the number of blocks available
            gaps[space_ind] = (gi + n_blocks, space)
        #
    return blocks



def checksum(blocks):
    """Computes the checksum for the blocks"""
    res = sum(i*val for i, val in enumerate(blocks) if val is not None)
    return res    
        

def solve(data: str) -> tuple[int|str, int|str]:

    disk_map = parse(data)
    blocks = files_as_blocks(disk_map=disk_map)

    compressed = compress(blocks=blocks)
    star1 = checksum(compressed)
    print(f"Solution to part 1: {star1}")
    
    compressed_no_defrag = compress(blocks=blocks, allow_fragmentation=False)
    star2 = checksum(compressed_no_defrag)
    
    
    print(f"Solution to part 2: {star2}")
    
    return star1, star2


def main() -> None:
    year, day = 2024, 9
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
