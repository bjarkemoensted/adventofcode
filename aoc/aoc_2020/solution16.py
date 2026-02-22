# ··• · ``. *· .   ·.+ `· *  `.·* ·   *  . `· . · `+.      ·   * * .· ·  .` · .·
# .  `   *·   *·    ·`+. ` *· · Ticket Translation  *.   .·. ·   .·*`   •` ·.·*+
#  *·. ·+     ·  ` *   https://adventofcode.com/2020/day/16 ·.*      .`·  *.  ·`
#  `*·  .·•+.`   ·.`*· •·. `*·  · `  `.+·   *      ·  ·+       ·.   *·  · ·* .`.

from dataclasses import dataclass
from functools import reduce

import numpy as np
from numpy.typing import NDArray


@dataclass()
class Constraint:
    label: str
    bounds: tuple[tuple[int, int], ...]

    def __init__(self, label: str, *bounds: tuple[int, int]) -> None:
        self.label = label
        self.bounds = tuple(sorted(bounds))
        
        for i, (a, b) in enumerate(self.bounds):
            assert b > a
            if i != 0:
                assert a > self.bounds[i-1][1]
            #
        #
    
    def __contains__(self, n: int) -> bool:
        return any(a <= n <= b for a, b in self.bounds)


def parse(s: str) -> tuple[list[Constraint], list[int], NDArray[np.int_]]:
    constraint_str, your_ticket_str, nearby_tickets_str = s.split("\n\n")
    
    constraints: list[Constraint] = []

    for line in constraint_str.splitlines():
        label, rangepart = line.split(": ")
        bounds = ((int(a), int(b)) for a, b in (part.split("-") for part in rangepart.split(" or ")))
        constraints.append(Constraint(label, *bounds))

    my_ticket = list(map(int, your_ticket_str.splitlines()[-1].split(",")))
    
    nearby_tickets = np.array([list(map(int, line.split(","))) for line in nearby_tickets_str.splitlines()[1:]])
    
    return constraints, my_ticket, nearby_tickets


def determine_validity(tickets: NDArray[np.int_], constraints: list[Constraint]) -> NDArray[np.bool_]:
    """Takes an array of i tickets x j fields and a list of k constraints.
    Returns a 3D boolean array where each element indicates whether for ticket i, field j
    satisfies contraint k."""

    n_fields = len(tickets[0])
    assert all(len(ticket) == n_fields for ticket in tickets)
    
    shape = (*tickets.shape, len(constraints))
    res = np.full(shape, False, dtype=bool)
    
    for (i, j), n in np.ndenumerate(tickets):
        for k, constraint in enumerate(constraints):
            res[i, j, k] = int(n) in constraint
        #

    return res


def deduce_fields(validity: NDArray[np.bool_]) -> list[int]:
    """Takes a 3D validity matrix and returns a list in which element
    i denotes the field corresponding to constraint i."""
    
    # Keep only validity data for tickets that aren't invalid
    valid_inds = np.where(validity.any(axis=-1).all(axis=-1))
    valid_tickets = validity[*valid_inds]

    # Compute a new boolean matrix, representing field-constraint compatibility
    _, n_fields, n_cons = valid_tickets.shape
    candidates = np.full((n_fields, n_cons), False, dtype=bool)
    
    for k in range(n_cons):
        for j in range(n_fields):
            # Constraint k might represent field j iff all values are valid
            candidates[j, k] = valid_tickets[:, j, k].all()

    # entry i is the correct field index of constraint i
    res = [-1 for _ in range(n_cons)]

    # Repeated method of exclusion, keep linking constraints to their only compatible field
    for _ in range(n_cons):
        # Find the indices of the constraint and field which may now be unambiguously linked
        assert sum((candidates[:, k].sum() == 1) for k in range(n_cons))
        compatible = ((k, [int(j) for meh in np.where(candidates[:, k]) for j in meh]) for k in range(n_cons))
        conind, fieldin = next((k, vals[0]) for k, vals in compatible if len(vals) == 1)
        
        # Add to results, and remove this field from candidates
        res[conind] = fieldin
        candidates[fieldin, :] = False

    return res


def solve(data: str) -> tuple[int|str, ...]:
    constraints, my_ticket, nearby_tickets= parse(data)

    # Determine validity between all ticket x field x constraint combinations
    validity = determine_validity(constraints=constraints, tickets=nearby_tickets)

    # Sum all the invalid fields (incompatible with all constraints)
    valid_mask = validity.any(axis=-1)
    star1 = nearby_tickets[np.where(~valid_mask)].sum()
    print(f"Solution to part 1: {star1}")

    # Link the constraint indices to the ticket indices
    reordered = deduce_fields(validity)
    # Get all the entries on the ticket which correspond to the departure fields
    departure_cons = (i for i, con in enumerate(constraints) if con.label.startswith("departure"))
    departure_inds = (my_ticket[reordered[i]] for i in departure_cons)

    star2 = reduce(lambda a, b: a*b, departure_inds, 1)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 16
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
