#   .·** ·    `·.  ·`*           ·`   · * . .·`     · .  * ·.+  `*  ·· .   ·`•·.
#  `·. ·* + · .  · .    * `·+  ·.   LAN Party ·       ·.   +·`  ·.· *  `*   ·   
# *·  . · `+. * · ·* . https://adventofcode.com/2024/day/23 `·  .· *      *`•· .
# .*`· .·   +· *  . ·`      ·  . `*.·  ` ·*+ . *`· . *  · `* .·    .   *· · *`.·


from itertools import combinations
import networkx as nx


def parse(s: str):
    res = [line.split("-") for line in s.splitlines()]
    return res


def count_candidate_cliques(cliques: list, size=3, must_start_with="t"):
    """Takes a list of cliques (sets of fully interconnected nodes) in a graph.
    Returns the number of distinct cliques with the specified size, starting with the specified string."""

    candidates = set()
    for clique in cliques:
        if len(clique) < size:
            continue  # Ignore cliques smaller than the target size
        
        # Otherwise, get all cliques and sub-cliques
        for comb in combinations(clique, size):
            if any(u.startswith(must_start_with) for u in comb):
                cand = tuple(sorted(comb))
                candidates.add(cand)
            #
        #
    
    res = len(candidates)
    return res
            

def solve(data: str) -> tuple[int|str, int|str]:
    links = parse(data)
    G = nx.Graph()
    G.add_edges_from(links)
    
    cliques = list(nx.find_cliques(G))
    star1 = count_candidate_cliques(cliques)
    print(f"Solution to part 1: {star1}")

    largest_clique = max((c for c in cliques), key=len)
    star2 = ",".join(sorted(largest_clique))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()