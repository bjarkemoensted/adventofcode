# `·. `+· . ·  `·.*+· •.` +   ·`  ·*.    ··* ·  `. • `   `·*  . · `·.+ .` ·.`·+*
#  . .·· ` *.`+   ·`     ` ·+. Space Stoichiometry   *·`.· ` .·`  *+·. · . ·* .`
# .`·`+ .*··  `.+   .· https://adventofcode.com/2019/day/14  ·   *·. ·` +*·  `·.
# `· ·.`   *`·· .*`. + ` ·· `  .   `··*+·`  *.   · .`   ·* .· ` *·.  `.*·  `·.`·

import math
from collections import Counter
from dataclasses import dataclass

import networkx as nx


@dataclass
class Reaction:
    """Represents a chemical reaction"""

    output_chemical: str  # The resulting chemical
    output_amount: int  # Amount of resulting chem
    requires: tuple[tuple[str, int], ...]  # tuples of (ingredient, amount)


def parse(s: str) -> list[Reaction]:
    res = []
    for line in s.splitlines():
        a, b = line.split(" => ")
        a_parts = a.split(", ")
        elems = [(chem, int(n)) for n, chem in (part.split() for part in a_parts + [b])]
        output_chem, output_amount = elems.pop()
        reqs = tuple(elems)
        reaction = Reaction(
            output_chemical=output_chem,
            output_amount=output_amount,
            requires=reqs
        )
        res.append(reaction)
        
    return res


class ReactionGraph:
    """Stores the chemical reactions and maintains a direted graph structure where
    chemicals are linked based on which chemical produces which."""

    SOURCE = "ORE"
    TARGET = "FUEL"

    def __init__(self, *reactions: Reaction) -> None:
        self.reactions = tuple(reactions)
        self.reactions_by_output = {r.output_chemical: r for r in self.reactions}
        # Make sure each chemical is produced by one distinct reaction
        if len(self.reactions_by_output) != len(self.reactions):
            raise RuntimeError
        
        # Store a graph where u -> v means u is involved in the reaction which produces v
        self.G = nx.DiGraph()
        for r in self.reactions:
            for req, _ in r.requires:
                self.G.add_edge(req, r.output_chemical, reaction=r)
            #
        
        # Reverse the graph so u -> v means to get u we need a reaction involving v
        self.G_rev = self.G.reverse()
        assert nx.is_directed_acyclic_graph(self.G_rev)
        # For each chemical, determine the set of chems we need (directly or indirectly) to produce it
        self.depends_on = {chem: nx.descendants(self.G_rev, chem) for chem in self.G_rev.nodes()}

    def min_cost(self, n=1) -> int:
        """Compute the minimum possible amount of source chemical required to produce
        amount n of the target chemical.
        The 'easy' error to make is to round up amounts when determining which reactions
        to use. This method keeps track of which ingredients among the requirements
        are themselves ingredients to other requirements. By only resolving ingredients
        which will not be used for producing any other ingredients, the rounding problem
        disappears - rounding up in these cases is not a problem, because there will
        be no point in the remaining calculations where we might encounter more of
        the ingredient in question."""

        # Keep this updated with the minimum of each ressource needed
        req = Counter({self.TARGET: n})

        while any(k != self.SOURCE for k in req.keys()):
            # Find the chems which aren't ingredients to any reactions we'll see from now on
            all_dependencies = set().union(*(self.depends_on[kp] for kp in req.keys()))
            independent = set(req.keys()) - all_dependencies
            if not independent:
                raise RuntimeError

            # Use those to determine the amount of each ingredient required
            for expand in sorted(independent):
                # Figure out the number of reactions required to produce the required amount
                amount_needed = req.pop(expand)
                reaction = self.reactions_by_output[expand]
                n_reactions = math.ceil(amount_needed / reaction.output_amount)

                # Update requirements
                for chem, n_in in reaction.requires:
                    req[chem] += n_in*n_reactions
                #
            #
        
        assert list(req.keys())[0] == self.SOURCE
        cost = sum(req.values())
        return cost
    
    def max_production(self, n_input: int, lower_bound=0) -> int:
        """Determines the max amount of target chemical which can be produced from amount n
        of the input chemical. A known initial lower bound can be provided as an optional parameter.
        This method works by using bijection search to narrow down a search region, until only
        a single value remains."""

        # Lower bound: Greatest amount known to be producible
        low = lower_bound
        assert self.min_cost(low) <= n_input
        
        # Upper bound: Smallest amount known not to be producible
        inc = 1
        high = low + inc

        # Grow the upper bound exponentially, until we find an amount we can't afford
        while self.min_cost(high) <= n_input:
            inc *= 2
            low = high
            high = low + inc

        while (diff := high - low) > 1:
            # Check if we can afford an amount halfway between the extremes
            midpoint = low + (diff // 2)
            affordable = self.min_cost(midpoint) <= n_input
            # Update the bounds
            if affordable:
                low = midpoint
            else:
                high = midpoint
            #

        # low is producible and high = low+1 is not, so low is the max
        return low


def solve(data: str) -> tuple[int|str, ...]:
    reactions = parse(data)

    G = ReactionGraph(*reactions)
    star1 = G.min_cost()
    print(f"Solution to part 1: {star1}")

    n_ore = 1_000_000_000_000
    # Use an integer number of the reactions from part 1 as a lower bound
    lower_fuel_bound = n_ore // star1
    star2 = G.max_production(n_input=n_ore, lower_bound=lower_fuel_bound)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 14
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
