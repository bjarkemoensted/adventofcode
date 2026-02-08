# `· `+.·   •   `   ·  *.·` ·`+.. •·   ·  `.· +`·. *.   ·` .·      *··` .+   ·`*
#   · . +·  .`·* `.  · · .*·`    Monster Messages  .·  ·`*.  • · ·. `*    `·.+*·
#   .··`     ·.  * ·`  https://adventofcode.com/2020/day/19   ·+  *·. · ` .··`. 
# ·`*  · .·  . ·` * .`    · *  + `  .·*· `  . ·  + ·` ·  .+  ·. *.`    ` ··.  ·`

from collections import defaultdict
from copy import deepcopy
from functools import cache
from itertools import product
from typing import TypeAlias

prodtype: TypeAlias = tuple[int|str, ...]
grammartype: TypeAlias = dict[int, list[prodtype]]


def parse(s: str) -> tuple[grammartype, list[str]]:
    rulespart, messagepart = s.split("\n\n")
    messages = messagepart.splitlines()

    grammar: grammartype = dict()

    for line in rulespart.splitlines():
        a, b = line.replace('"', '').split(": ")
        k = int(a)
        assert k not in grammar
        productions = [tuple(int(elem) if elem.isdigit() else elem for elem in part.split()) for part in b.split("|")]
        grammar[k] = productions
    
    return grammar, messages


def to_cnf(grammar: grammartype) -> grammartype:
    """Checks whether the input rules are in Chomsky Normal Form. Raises RuntimeError if not."""
    
    res = deepcopy(grammar)

    all_prods = [prod for rule in res.values() for prod in rule]
    # START: Ensure the start symbol is absent from all right hand sides
    assert not any(0 in prod for prod in all_prods)

    # TERM: Check no productions mix terminals and non-terminals
    assert all(any(all(isinstance(elem, type_) for elem in prod) for type_ in (str, int)) for prod in all_prods)

    max_ = max(res.keys())

    # BIN: expand productions of more than 2 nonterminals
    for k, productions in list(res.items()):
        for i, prod in enumerate(productions):
            if len(prod) <= 2:
                continue

            max_ += 1
            res[k][i] = (prod[0], max_)
            rem = prod[1:]
            while len(rem) > 2:
                res[max_] = [(rem[0], max_+1)]
                max_ += 1
                rem = rem[1:]
            
            res[max_] = [rem]
            

    def _is_unit(prod: prodtype) -> bool:
        return len(prod) == 1 and isinstance(prod[0], int)

    # UNIT: Eliminate unit rules like A: B, B: C -> A: C 
    for k, productions in list(res.items()):
        for i in reversed(range(len(productions))):
            prod = productions[i]
            
            if not _is_unit(prod):
                continue
            
            subs = set()
            seed = prod[0]
            assert isinstance(seed, int)
            front: set[int|str] = {seed}

            while front:
                next_ = list(front)
                front = set()
                for other in next_:
                    assert isinstance(other, int)
                    for p in res[other]:
                        if _is_unit(p):
                            front.add(p[0])
                        else:
                            subs.add(p)
                        #
                    #
            
            del res[k][i]
            res[k] += list(subs)
        #

    pointers = (x for prod in res.values() for p in prod for x in p if isinstance(x, int))
    assert all(p in res for p in pointers)

    return res


class Parser:
    def __init__(self, grammar: grammartype) -> None:
        self.grammar = to_cnf(grammar)
        self.cache: dict[str, set[tuple[int, ...]]] = dict()
        # Construct an inverse mapping, from productions to the heads
        self.inv: dict[prodtype, set[int]] = defaultdict(set)
        for k, v in self.grammar.items():
            for prod in v:
                self.inv[prod].add(k)
            #
        #
    
    @cache
    def _get_predecessors(self, left: frozenset, right: frozenset) -> set:
        """Takes two sets of elements of the grammar. Returns the set of all combinations
        of rules which could have produced the two sides."""
        res = set.union(*map(self.inv.__getitem__, product(left, right)))
        
        return res

    def is_valid(self, message: str) -> bool:
        """Determines whether the input string is valid given the parser's grammar"""
        n = len(message)
        table: list[list[frozenset[int]]] = [[frozenset() for _ in range(n)] for _ in range(n)]

        # First step: note which rules can produce each of the characters in the final string
        for i in range(n):
            char = message[i]
            for j, rule in self.grammar.items():
                for production in rule:
                    if (char,) == production:
                        table[0][i] = frozenset({j}) | table[0][i]
                    #
                #
            #

        # DP step: break into substrings of varying lengths and shifts, noting which rules produce each
        for i in range(1, n):  # substring length
            for j in range(n-i):  # window start
                for k in range(i):
                    cut = k + 1
                    left = table[k][j]
                    right = table[i-cut][j+cut]
                    # If either side cannot be produced, there's no combinations
                    if not left or not right:
                        continue

                    # Find all possible ways to produce the left and right substrings
                    predecessors = self._get_predecessors(left, right)
                    table[i][j] |= predecessors
                #
            #

        # Check if the starting rule S (index 0 here) produces the message
        initial = table[n-1][0]
        res = 0 in initial
        return res


def solve(data: str) -> tuple[int|str, ...]:
    
    grammar, messages = parse(data)
    parser = Parser(grammar)

    star1 = sum(parser.is_valid(m) for m in messages)
    print(f"Solution to part 1: {star1}")

    # Update the grammar for part 2 and make a new parser
    grammar[8] = [(42,), (42, 8)]
    grammar[11] = [(42, 31), (42, 11, 31)]
    parser2 = Parser(grammar)

    star2 = sum(parser2.is_valid(m) for m in messages)

    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 19
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
