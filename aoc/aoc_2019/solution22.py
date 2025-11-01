# .·· *   ·.·` ·        ·.·`*  · `·   ·`*.· +    *`.·  ·`+ .·  ·  `.*  ·· · `·.`
# ·.`.·· ·* `·   ` · *. *·   ·` ·* Slam Shuffle ·.+`    · · *· ` ·. `*·    ·•·`*
# •·.·`.   * *·   .`·  https://adventofcode.com/2019/day/22   .·* ·`  . `· `·*·.
# ·`*  •· ·  .`·* `.*· ·`   ·+.  ·.*·   `· · .+  · ·.  *. `·  ·   *·· `  *·· ..·

from __future__ import annotations

from collections import deque
from copy import deepcopy
from functools import singledispatchmethod
from typing import Literal, TypeAlias, TypeVar, cast

import sympy

deck: TypeAlias = deque[int]
# Operation - one of the shuffling operations
optype: TypeAlias = Literal["new_stack", "cut", "deal_with_increment"]
# Instruction - an operation along with a tuple of any arguments
instype: TypeAlias = tuple[optype, tuple[int, ...]]
# Valid type for instantiating a modular scalar
modtype: TypeAlias = int|sympy.Basic


def parse(s: str) -> list[instype]:
    res: list[instype] = []
    for line in s.splitlines():
        if line == "deal into new stack":
            res.append(("new_stack", ()))
        elif line.startswith("cut"):
            res.append(("cut", (int(line.split()[-1]),)))
        elif line.startswith("deal"):
            res.append(("deal_with_increment", (int(line.split()[-1]),)))
        #

    print(res)
    return res


T = TypeVar('T', int, deque[int], sympy.Basic)


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
        
        raise NotImplementedError(f"No function registered for type {type(obj)}")
    
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
    
    @apply_operation.register
    def _(self, obj: sympy.Basic, operation: optype, args: tuple[int, ...]):
        """Apply shuffling operation on an algebraic symbol for an index"""
        
        match operation:
            case "new_stack":
                return -1 - obj
            case "cut":
                n = args[0]
                return (obj - n)
            case "deal_with_increment":
                n = args[0]
                return (obj*n)
            case _:
                raise RuntimeError(f"Unrecognized operation: {operation}")
    
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
                revmod = pow(n, -1, mod=self.n_cards)
                return (operation, (revmod,))
            case _:
                raise RuntimeError(f"Unrecognized operation: {operation}")
            #
        #
    
    def reverse_instructions(self, instructions: list[instype]) -> list[instype]:
        """Reverse multiple instructions"""
        res = [self.reverse_instruction(ins) for ins in reversed(instructions)]
        return res
    
    def determine_recurrence_relation_coefficients(self, instructions: list[instype]) -> tuple[int, int]:
        """Shuffling follows a recurrence relation where X_n+1 is defined in terms of X_n as
        X_n+1 = a + b*X_n
        
        Here, the n'th term can be shown to be expressed in terms of X_0 as
        b^n + a*sum_i=0^n-1 b^i
        Using the formula for finite geometric series sum_i=0^n-1 b^i = (1 - b^n)/(1 - b), the expression simplifies to
        a/(1 - b) + (X_0 - a/(1-b)*b^n).
        This method returns the values of a and b for the recurrence corresponding to the input instructions."""
        
        x_sym = sympy.Symbol("x")
    
        subsequent_expr = self._apply_instructions(instructions, obj=x_sym)
        coeffs = subsequent_expr.as_coefficients_dict()
        a, b = map(int, (coeffs[k] for k in (1, x_sym)))
        assert all(isinstance(val, int) for val in (a, b))
        
        return a, b
    
    def final_index(self, index_start: int, instructions: list[instype], reverse: bool=False, n: int=1) -> int:
        """Computes the final position of the card which is initially at index_start, after applying
        the shuffling instructions n times.
        If reverse is True, the computation is based instead on the inverse of the provided shuffling instructions,
        so the result indicates the original position of a card which ends up at index_start
        after shuffling n times."""
        
        # If reverse, compute the inverse shuffling
        if reverse:
            instructions = self.reverse_instructions(instructions)
        
        # Determine the coefficients of the recurrence relation x -> a + b*x for the shuffling.
        a, b = self.determine_recurrence_relation_coefficients(instructions=instructions)
        
        # The recurrence is a geometric series which evaluates to (x0 - f)*b^n + f, where f = a/(1-b). bc math
        
        # Compute f = a/(1-b) as a times the modular inverse of (1 - b)
        k = self.n_cards
        modinv = pow(1 - b, -1, k)  # this determines the modular inverse, raising an error if 1-b and k aren't coprime
        f = a*modinv
        
        # Compute b^n mod k (python uses an efficient modular exponentiation algo under the hood here)
        b_pow_n_mod_k = pow(base=int(b), exp=n, mod=k)
        
        # Evaluate the expression for the geometric series modulo n_cards. This is the card's final location.
        res = ((index_start - f)*b_pow_n_mod_k + f) % k
        return res
    
    def __call__(self, instructions: list[instype], obj: T|None=None, reverse: bool=False, n_times: int = 1) -> T:
        """Apply one or more shuffling instructions to the specified index/deck.
        if reverse is True, applies the inverse of the instructions."""
        
        # If we're operating on a single index, compute the final idex analylitically
        if isinstance(obj, int):
            return self.final_index(index_start=obj, instructions=instructions, reverse=reverse, n=n_times)
        
        if reverse:
            instructions = self.reverse_instructions(instructions=instructions)
        
        # If object isn't specified, create a fresh deck of cards to shuffle
        new_obj = obj if obj is not None else cast(T, self.new_deck())
        
        # Otherwise, repeatedly shuffle until we're done
        new_obj = deepcopy(new_obj)
        for _ in range(n_times):
            new_obj = self._apply_instructions(instructions=instructions, obj=new_obj)
        
        return new_obj
    

def solve(data: str) -> tuple[int|str, int|str]:
    instructions = parse(data)
    n_cards = 10007
    shuffler = Shuffler(n_cards=n_cards)
    
    start = 2019
    star1 = shuffler(instructions, obj=start)
    print(f"Solution to part 1: {star1}")

    n_large = 119_315_717_514_047
    n_shuffles = 101_741_582_076_661
    shuffler2 = Shuffler(n_cards=n_large)
    target = 2020
    
    star2 = shuffler2(instructions, target, reverse=True, n_times=n_shuffles)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 22
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
