# `ꞏ. .ꞏ*⸳` .⸳  *`     *   +.⸳ꞏ ` + ⸳  .ꞏ⸳`+   `  +.⸳•⸳`   .*  ꞏ.` ⸳* ꞏ ⸳. *• ⸳ꞏ
# *.`+ ⸳ ꞏ*     .   ⸳` ꞏ+  `ꞏ *⸳ . Slam Shuffle  ⸳` *`.⸳+    ꞏ*   ⸳ .*⸳ꞏ   . ⸳` 
# .⸳+`ꞏ ⸳`.  +  ⸳•*` . https://adventofcode.com/2019/day/22  +.   `• ꞏ   *.⸳ꞏ• .
# `*      ꞏ ꞏ`⸳+ +.⸳ꞏ⸳   .  ⸳•`  ꞏ *. • +ꞏ ` .       ` *ꞏ⸳ *⸳.  `+ꞏ*      ꞏ`.` ⸳

from abc import abstractmethod, ABC
from collections import deque
from copy import deepcopy
from functools import singledispatchmethod
from typing import cast, Generic, get_args, Literal, TypeAlias, TypeVar

deck: TypeAlias = deque[int]
# Operation - one of the shuffling operations
optype: TypeAlias = Literal["new_stack", "cut", "deal_with_increment"]
# Instruction - an operation along with a tuple of any arguments
instype: TypeAlias = tuple[optype, tuple[int, ...]]


test = """deal into new stack
cut -2
deal with increment 7
cut 8
cut -4
deal with increment 7
cut 3
deal with increment 9
deal with increment 3
cut -1"""


def parse(s):
    res = []
    for line in s.splitlines():
        if line == "deal into new stack":
            elem = ("new_stack", ())
        elif line.startswith("cut"):
            elem = ("cut", (int(line.split()[-1]),))
        elif line.startswith("deal"):
            elem = ("deal_with_increment", (int(line.split()[-1]),))
        res.append(elem)
    return res


def modinv(n, N) -> int:
    """Uses Euclid's extended algorithm to determine the inverse module number for n, given N.
    This is only well-defined if n and N are co-prime (otherwise, the mapping i -> (i*n)%N) is not bijective).
    If not, an error is raised."""
    
    t, new_t = 0, 1
    r, new_r = N, n
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        raise ValueError(f"{n} has no inverse mod {N}")
    
    return t % N


# TODO should probably add sympy vars here if we end up doing that...
T = TypeVar('T', int, deque[int])


class Shuffler:
    """For a specified deck size, handles applying the various shuffling operations,
    and exposes helper methods for shuffling a full deck (deque, heheh) of cards, 
    or just computing the new location of a specific index after applying one or more such operations.
    Also exposes methods for reversing shuffling operations, allowing one to trace an index backwards
    through one or more shuffling operations."""

    def __init__(self, n_cards: int):
        self.n_cards = n_cards
    
    def new_deck(self) -> deque[int]:
        """Provides a fresh deck in 'factory order' (0, 1, 2, ...)."""
        res = deque(range(self.n_cards))
        return res
    
    def _apply_instructions(self, instructions: list[instype], obj: T) -> T:
        """Applies a series of instructions to the target object (obj), which can represent
        either a full deck of cards, or an index."""
        
        res = obj
        for operation, args in instructions:
            res = self.apply_operation(res, operation, args)
        
        return res
        
    @singledispatchmethod
    def apply_operation(self, obj: T, operation: optype, args: tuple[int, ...]):
        """Dispatch for appying a shuffling operation - specific implementations are provided
        for full decks/indices"""
        raise NotImplementedError
    
    @apply_operation.register
    def _(self, obj: deque, operation: optype, args: tuple[int, ...]) -> deque[int]:
        """Apply a shuffling operation to a full deck. While somewhat inefficient, this takes a deep copy
        of the deck before doing anything, mainly to make debugging etc. easier. Since we can't
        efficiently operate on a full deck for large datasets anyway, this shouldn't matter much."""
        
        match operation:
            case "new_stack":
                # Dealing into a new stack simply reverses the card order
                res = deepcopy(obj)
                res.reverse()
            case "cut":
                # Cutting is just a left-rotation of the double ended queue
                res = deepcopy(obj)
                n = args[0]
                res.rotate(-n)
            case "deal_with_increment":
                # Deals cards to indices with increment n, module n_cards
                n = args[0]
                rearranged = [-1 for _ in obj]
                for i, card in enumerate(obj):
                    new_ind = (n*i) % len(rearranged)
                    # Make sure no card has been placed here yet
                    if rearranged[new_ind] != -1:
                        raise RuntimeError(f"Card has already been placed at index {new_ind}")
                    rearranged[new_ind] = card
                res = deque(rearranged)
            case _:
                raise RuntimeError(f"Unrecognized operation: {operation}")
            #
        return res

    @apply_operation.register
    def _(self, obj: int, operation: optype, args: tuple[int, ...]):
        """Apply shuffling operation on an index"""
        match operation:
            case "new_stack":
                return -1 - obj
            case "cut":
                n = args[0]
                return (obj - n) % self.n_cards
            case "deal_with_increment":
                n = args[0]
                return (obj*n) % self.n_cards
            case _:
                raise RuntimeError(f"Unrecognized operation: {operation}")
            #
    
    def reverse_instruction(self, instruction: instype) -> instype:
        """Computes the inverse of a shuffling operation"""
        
        operation, args = instruction
        match operation:
            case "new_stack":
                # Reversing the order is its own inverse, so nothing to do here
                return instruction
            case "cut":
                # Cutting/left rotation is just the inverse of rotating the other direction
                n = args[0]
                return (operation, (-n,))
            case "deal_with_increment":
                # Here, we need the inverse modulo of n, for the given n_cards.
                n = args[0]
                # Find the correct modulo using Euclid's extended algorithm.
                revmod = modinv(n, self.n_cards)
                return (operation, (revmod,))
            case _:
                raise RuntimeError(f"Unrecognized operation: {operation}")
            #
        #
    
    def reverse_instructions(self, instructions: list[instype]) -> list[instype]:
        """Reverse multiple instructions"""
        res = [self.reverse_instruction(ins) for ins in reversed(instructions)]
        return res
    
    def __call__(self, instructions: instype|list[instype], obj: T|None=None, reverse: bool=False) -> T:
        """Apply one or more shuffling instructions to the specified index/deck.
        if reverse is True, applies the inverse of the instructions."""
        
        if not isinstance(instructions, list):
            instructions = [instructions]
        if reverse:
            instructions = self.reverse_instructions(instructions=instructions)
        
        if obj is None:
            obj = cast(T, self.new_deck())
        
        res = self._apply_instructions(instructions=instructions, obj=obj)
        return res



def solve(data: str):
    instructions = parse(data)
    n_cards = 10007
    shuffler = Shuffler(n_cards=n_cards)
    
    star1 = shuffler(instructions, obj=2019)
    print(f"Solution to part 1: {star1}")
    
    n_large = 119_315_717_514_047
    n_shuffles = 101_741_582_076_661

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2019, 22
    from aocd import get_data
    raw = get_data(year=year, day=day)
    #raw = test
    solve(raw)


if __name__ == '__main__':
    main()
