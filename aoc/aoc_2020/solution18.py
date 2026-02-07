# ·· +·`·.•  · * ` .· `·* ·. +· ` .· *  ·`  ·  ·.·  ·.*  ·+  `·.    · ·•+ `· .·*
# .`· .·   *+· .·*  •· . ·.· ` + Operation Order .· ` ·   ·*.· `*    .•· `·* ··.
#  ·*. +  `·`.··  `+ * https://adventofcode.com/2020/day/18 *  · `·•.   `·* ·` `
# ·.`*·.• ·.· . `·*  ·.· .  `·•· `+ · ·. ·  ·  `     ·.+` ` ·*.    · ·* ·  . · ·

import operator
from typing import Iterator

ops = {
    "+": operator.add,
    "*": operator.mul
}


def parse(s: str) -> list[str]:
    return s.splitlines()


def extract_tokens(s: str) -> Iterator[str|int]:
    buffer = ""
    for char in s:
        if char.isdigit():
            buffer += char
        else:
            if buffer:
                yield int(buffer)
                buffer = ""
            if char != " ":
                yield char
            #
        #
    if buffer:
        yield int(buffer)
    #


def shunting_yard(expr: str, precedence: list[str]) -> list[int|str]:
    """Uses Dijkstra's shunting yard algorithm to convert the input expression, using the input precedence,
    into reverse Polish notation.
    expr is the expression in normal (infix) notation.
    precedence is a list of iterables of strings representing the order of operations. For example,
    ["+-", "*/"] for normal precedenc. """

    # Unpack the precedence, e.g. ["+-", "*/"] => {"+": 0, ...}
    order = {symbol: i for i, symbols in enumerate(precedence) for symbol in symbols}
    
    output: list[int|str] = []
    ops: list[str] = []

    for elem in extract_tokens(expr):
        # Add numbers to output
        if isinstance(elem, int):
            output.append(elem)
            continue
        
        assert isinstance(elem, str)
        
        is_operator = elem in order
        if is_operator:
            # Pop higher-order operators from stack before proceeding (also pop same order, to eval left-to-right)
            _order = order[elem]
            while ops and order.get(ops[-1], -1) >= _order:
                output.append(ops.pop())
            # Add operator to stack
            ops.append(elem)
            continue
        elif elem == "(":
            ops.append(elem)
        elif elem == ")":
            while ops:
                # Pop operators from stack until we encounter a matching parenthesis
                other = ops.pop()
                if other == "(":
                    break
                else:
                    output.append(other)
                #
            # 
        else:
            raise RuntimeError
        #
    
    while ops:
        output.append(ops.pop())
    return output


def evaluate(expr: str, precedence: list[str]) -> int:
    """Evaluates an expression, using the input order of operations"""
    
    # Convert into reverse Polish notation
    rpn = shunting_yard(expr, precedence)

    # Evaluate the rpn expression
    stack: list[int] = []
    for elem in rpn:
        if isinstance(elem, int):
            stack.append(elem)
        else:
            op = ops[elem]
            right = stack.pop()
            left = stack.pop()
            stack.append(op(left, right))
        #

    res = stack.pop()
    assert not stack
    return res


def solve(data: str) -> tuple[int|str, ...]:
    expressions = parse(data)
    precedence = ["*+"]

    star1 = sum(evaluate(expr, precedence) for expr in expressions)
    print(f"Solution to part 1: {star1}")

    precedence = ["*", "+"]
    star2 = sum(evaluate(expr, precedence) for expr in expressions)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 18
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
