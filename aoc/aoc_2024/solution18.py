# .`*  ꞏ.  `*     ꞏ`.⸳    * ` . ` •⸳ꞏ    ꞏ  . `*ꞏ⸳   +ꞏ    ⸳ꞏ*   `ꞏ+    ` ꞏ• ⸳.ꞏ
#  ꞏ`*ꞏ  ⸳+   ꞏ`. • ⸳ꞏ        •`⸳ `  RAM Run ꞏ   +  ` *.ꞏ  .•   ꞏ  ` *⸳`   ꞏ .+ 
# `ꞏ  ⸳ ꞏ*  `⸳  • .ꞏ   https://adventofcode.com/2024/day/18   `  . . ⸳ ꞏ  ꞏ `` *
#  + .*⸳`ꞏ   .       + ꞏ ` ꞏ ` •*⸳   ꞏ* ` .ꞏ ` .*ꞏ        .  ꞏ• ⸳ `       • • `.


import networkx as nx


def parse(s):
    res = [tuple(map(int, line.split(","))) for line in s.splitlines()]
    return res


class MemorySpace:
    """Represents the memory space from the puzzle. Keeps track of the falling bytes and
    their order, and offers functionality for pathfinding after a variable number of bytes
    have fallen."""
    
    def __init__(self, byte_coords: list):
        self.byte_coords = [(x, y) for x, y in byte_coords]
        
        # Set the shape (note xy_format, so x, y = j, i)
        x, y = zip(*byte_coords)
        assert all(min(coords) == 0 for coords in (x, y))
        self.shape_xy = max(x) + 1, max(y) + 1
    
    def _iter_xy(self):
        """Iterates over all (x,y)-tuples in the space"""
        cols, rows = self.shape_xy
        for y in range(rows):
            for x in range(cols):
                yield x, y
    
    def as_string(self, n_bytes: int):
        """Helper method for displaying the memspace after n fallen bytes as a string.
        Looks like the example in the puzzle, so can be helpful for debugging and stuff."""
        lines = []
        cols, rows = self.shape_xy
        for y in range(rows):
            lines.append("".join(["#" if (x, y) in self.byte_coords[:n_bytes] else "." for x in range(cols)]))
        
        res = "\n".join(lines)
        return res
    
    def graph_after__n_bytes(self, n: int):
        """Returns a graph in which nodes represent free coordinates (not blocked by falling bytes),
        and edges exist between adjacent free coordinates."""

        blocked = set(self.byte_coords[:n])
        G = nx.Graph()
        
        for u in self._iter_xy():
            if u in blocked:
                continue  # don't connect blocked coordinates to anything
            
            # Get the adjacent coordinates (graph i undirected, so only iterate in positive direction)
            x, y = u
            for v in ((x, y+1), (x+1, y)):
                # Skip out-of-bounds neighbors
                if not all(0 <= c < dim for c, dim in zip(v, self.shape_xy)):
                    continue
                # Connect u-v if v if not blocked by a fallen byte
                if v not in blocked:
                    G.add_edge(u, v)
                #
            #
        return G

    def shortest_path_after_n_bytes(self, n_bytes: int, start=(0, 0), target=None):
        """Returns the shortest path after n bytes have fallen.
        Start and target default to upper left and lower right corners, respectively."""

        if target is None:
            cols, rows = self.shape_xy
            target = (cols - 1, rows - 1)
        
        # Make the graph after n bytes have fallen, and compute shortest path
        G = self.graph_after__n_bytes(n=n_bytes)
        res = nx.shortest_path_length(G=G, source=start, target=target)
        
        return res
    
    def first_blocking_byte(self, start=(0, 0), target=None):
        """Returns the coordinate of the first byte which blocks off the path from start to target.
        Works by doing binary for the smallest number of fallen bytes such that no path exists.
        Start and target default to upper left and lower right corners, respectively."""

        if target is None:
            cols, rows = self.shape_xy
            target = (cols - 1, rows - 1)
        
        def is_blocked(n: int) -> bool:
            """Returns a bool indicating whether the path is blocked after n bytes"""
            G = self.graph_after__n_bytes(n=n)
            return not nx.has_path(G=G, source=start, target=target)
        
        left = 0  # greatest n_bytes where path still exists
        right = len(self.byte_coords)  # smallest n_bytes where no path exists
        
        # Check that there exists n where path is/isn't blocked
        assert not is_blocked(left)
        assert is_blocked(right)
        
        # Binary search for the n where the path becomes blocked
        while right - left > 1:
            mid = (right + left) // 2
            if is_blocked(mid):
                right = mid
            else:
                left = mid
            #
        
        # Block occurs after n bytes, so the blocking byte as at index n-1
        ind = right-1
        res = self.byte_coords[ind]
        
        return res
        

def solve(data: str):
    byte_coords = parse(data)    
    ms = MemorySpace(byte_coords=byte_coords)

    n_bytes_fall = 1024 if len(byte_coords) > 100 else 12  # 12 for sample data, 1024 for real
    star1 = ms.shortest_path_after_n_bytes(n_bytes=n_bytes_fall)
    print(f"Solution to part 1: {star1}")

    first_blocking_byte_coord = ms.first_blocking_byte()
    star2 = ",".join(map(str, first_blocking_byte_coord))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2024, 18
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
