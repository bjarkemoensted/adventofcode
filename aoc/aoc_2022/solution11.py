# .*· · `+·  `· *·    •.`  .·` .·` ·   .    *·` · .`   +·  .· `*   .·`·   *  .·`
# .· * . ·  `. ·    +· ` .     Monkey in the Middle .·   *.·`+   ·  .·`   . · +·
# ·`.·.   `·. * ·  ·`. https://adventofcode.com/2022/day/11 .·    ·*   .·   `·.·
# `.•`·· .  ··`.  * · .  ` `.*· .•  ·  · *   .· .`· + `·. ·*  ·  *`  . ·`  .*·` 

import math
import operator
from dataclasses import dataclass, field
from typing import Callable, Literal, Self, TypeGuard, get_args

type optype = Literal["+", "*"]
ops = {"+": operator.add, "*": operator.mul}


def is_op(obj) -> TypeGuard[optype]:
    return isinstance(obj, str) and obj in get_args(optype)


@dataclass
class Func:
    """Callable for updating a worry level"""
    lhs: int|None  # None for old worry level
    rhs: int|None
    op: Callable[[int, int], int]

    def __call__(self, val: int) -> int:
        a = val if self.lhs is None else self.lhs
        b = val if self.rhs is None else self.rhs
        res = self.op(a, b)
        return res
    #


@dataclass
class Monkey:
    id_: int
    starting_items: tuple[int, ...]
    operation: str
    worry_func: Callable[[int], int] = field(init=False)
    divisor: int
    pass_if_true: int
    pass_if_false: int

    def __post_init__(self) -> None:
        """Set up the function for determining recipient monkey"""
        expr = self.operation.split(" = ")[-1]
        lhs_str, op_str, rhs_str = expr.split()
        lhs = None if lhs_str == "old" else int(lhs_str)
        rhs = None if rhs_str == "old" else int(rhs_str)
        op = ops[op_str]
        func = Func(lhs=lhs, rhs=rhs, op=op)
        self.worry_func = func


def parse(s: str) -> list[Monkey]:
    """Parses puzzle input into a dictionary where keys are the monkey numbers
    and values are dicts representing the states of each monkey."""

    res = []
    chunks = s.split("\n\n")
    for chunk in chunks:
        lines = chunk.splitlines()
        id_ = int(lines[0].split("Monkey ")[-1][:-1])
        # List of items currently held by the monkey
        items = tuple(int(stuff) for stuff in lines[1].split("Starting items: ")[-1].split(", "))
        # Function to update the 'worry level' when the monkey examines objects
        operation = lines[2].split("Operation: new = ")[-1]
        # Get the divisor the monkey uses to determine to whom an items should be thrown
        divisor = int(lines[3].split("Test: divisible by ")[-1])

        # Recipient monkey if divisible/not divisible
        truepass = int(lines[4].split("If true: throw to monkey ")[-1])
        falsepass = int(lines[5].split("If false: throw to monkey ")[-1])

        monkey = Monkey(
            id_=id_,
            starting_items=items,
            divisor=divisor,
            operation=operation,
            pass_if_false=falsepass,
            pass_if_true=truepass
        )
        
        res.append(monkey)
        
    return res


class KeepAway:
    def __init__(self, *monkeys: Monkey, decrement_worry=True):
        self.monkeys = {monkey.id_: monkey for monkey in monkeys}
        self.items = {id_: list(monkey.starting_items) for id_, monkey in self.monkeys.items()}
        self.n_inspected = {id_: 0 for id_ in self.monkeys.keys()}
        self.decrement_worry = decrement_worry
        # Determine least common multiple of all the monkeys' divisors
        divs = [monkey.divisor for monkey in self.monkeys.values()]
        self.lcm = math.lcm(*divs)

    def _monkey_turn(self, monkey_id: int) -> None:
        """Let the monkey with the specified ID play its turn"""
        
        monkey = self.monkeys[monkey_id]
        while True:
            # Keep trying to throw the next item
            try:
                item = self.items[monkey_id].pop()
            except IndexError:
                return
            
            # Use the monkeys update function to determine my new worry level over the item
            new_worry = monkey.worry_func(item)
            self.n_inspected[monkey_id] += 1
 
            # Calm down a bit after inspection
            if self.decrement_worry:
                new_worry = new_worry // 3
            elif new_worry > self.lcm:
                # all divisors divide lcm, so any factor of that is irrelevant to monkey logic
                new_worry = self.lcm + new_worry % self.lcm

            # Throw the item to the next monkey
            target = (
                monkey.pass_if_true
                if new_worry % monkey.divisor == 0
                else monkey.pass_if_false
            )
            self.items[target].append(new_worry)
        #

    def tick(self, n_rounds: int=1) -> Self:
        """Play n rounds of the keep away game"""
        order = sorted(self.monkeys.keys())

        for _ in range(n_rounds):
            for n in order:
                self._monkey_turn(n)
            #
        return self
    
    def monkey_business(self) -> int:
        """Compute Monkey Business Score (product of two largest n items inspected)."""
        a, b = sorted(self.n_inspected.values(), reverse=True)[:2]
        res = a*b
        return res
    #


def solve(data: str) -> tuple[int|str, ...]:
    monkeys = parse(data)

    star1 = KeepAway(*monkeys).tick(n_rounds=20).monkey_business()
    print(f"Solution to part 1: {star1}")

    star2 = KeepAway(*monkeys, decrement_worry=False).tick(n_rounds=10_000).monkey_business()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 11
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
