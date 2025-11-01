# ·+`·.`    •.·   `· *   ·* `  .   ·  *   · .   *·`·.+  ` ·     .·`+. ·.  . `· *
# *.·`·.   * ·`*· .   .   ` ·`+• Memory Maneuver   . ·    .` *  ·•.      `·.·`*`
#  * .`  ·*.  .·  ·  ` https://adventofcode.com/2018/day/8 · ·  `.*    ·. `* .·.
# ·`. +·`.·.`  · `    +·.  * • `.·  .     .·*`·.     `  .  .·`* +`·. ·   ·*`.+`·

from __future__ import annotations

test = """2 3 0 3 10 11 12 1 1 0 1 99 2 1 1 2"""


def parse(s: str):
    return [int(c) for c in s.split()]


class Node:
    """Class for representing a node. Just to have functionality for setting parents/children in a simple way"""
    
    def __init__(self, parent:Node|None, metadata: list[int]|None=None):
        self.parent: Node|None = None
        self.metadata: list[int] = metadata if metadata is not None else []
        self.children: list[Node] = []
        if parent is not None:
            self.add_parent(parent)
    
    def add_parent(self, other: Node, reciprocal=True):
        """Sets input node as parent of this one. Also registers this node as child of the parent"""
        if self.parent is not None:
            raise RuntimeError
        self.parent = other
        if reciprocal:
            other.add_child(self, reciprocal=False)
        #
    
    def add_child(self, other: Node, reciprocal=True):
        """Add a child node. Set this node as parent of the child"""
        self.children.append(other)
        if reciprocal:
            other.add_parent(self, reciprocal=False)
        #
    
    def recsum(self) -> int:
        """Compute recursive sum of all metadata entries"""
        res = sum(self.metadata)
        for child in self.children:
            res += child.recsum()
        
        return res
    
    def value(self) -> int:
        """Compute the nodes 'value' - sum of metadata if no children, sum of value of children
        with indices given by metadata (index 1)."""
        
        if not self.children:
            return sum(self.metadata)
        
        res = 0
        for i in self.metadata:
            # Ignore out of bound indices
            try:
                res += self.children[i-1].value()
            except IndexError:
                pass
            #
        return res


def parse_tree(data, parent=None, i=0) -> tuple[Node, int]:
    """Parses data into a tree structure.
    Returns a Node, and the current index while parsing (this is to make the recursive step simpler).
    The initial call will return the root node of the tree."""
    
    # Use next two indices to parse number of children and number of metadata entries
    n_children, n_meta = data[i], data[i+1]
    node = Node(parent=parent)
    i += 2
    
    # Parse the children before completing
    for _ in range(n_children):
        _, i = parse_tree(data, parent=node, i=i)
    
    # After the children, read the metadata
    node.metadata += data[i:i+n_meta]
    i += n_meta
    
    return node, i


def solve(data: str) -> tuple[int|str, int|str]:
    parsed = parse(data)
    tree, _ = parse_tree(parsed)
    
    star1 = tree.recsum()
    print(f"Solution to part 1: {star1}")

    star2 = tree.value()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 8
    from aocd import get_data
    
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
