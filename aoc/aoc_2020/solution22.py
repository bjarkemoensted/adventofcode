# ·.` ·•.·*  `·  · *.   ·   `·+  ·`  •` ·. ·  + *··  .• ·  *· ·`.* · `·+   · .·*
#  ··`+ ·`.  * · `.  ·*  +  ·`   . Crab Combat ·`*. ··    `+   · `.* ·.`*· `.•·.
#  . ·` *· ·`.  •   •  https://adventofcode.com/2020/day/22    •·.` ·   `*.· · `
# ·`.+ ·  ` ·  .·.+`·  * ·   +`.·  .`  *`+·     ·`. *·.  ·*      ·*`   ·*.· ·`+·

from collections import deque


def parse(s: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    deck_strings = (part.splitlines()[1:] for part in s.split("\n\n"))
    
    decks = (tuple(map(int, vals)) for vals in deck_strings)
    a, b = (ds for ds in decks)
    return a, b


def compute_score(*cards: int) -> int:
    res = sum((i+1)*c for i, c in enumerate(reversed(cards)))
    return res


def play_game(player_deck_tup: tuple[int, ...], opponent_deck_tup: tuple[int, ...], with_recursion=False) -> int:
    """Plays a game of (recursive) combat.
    Returns the score resulting from the game (positive if player wins, negative if opponent wins)."""
    
    assert isinstance(player_deck_tup, tuple) and isinstance(opponent_deck_tup, tuple)

    # Set up decks
    player_deck = deque(player_deck_tup)
    opponent_deck = deque(opponent_deck_tup)
    
    # Keep a history of seen states, to avoid infinite loops
    history: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    while player_deck and opponent_deck:
        # Detect recurrences to avoid infinite loops
        state = (tuple(player_deck), tuple(opponent_deck))
        if state in history:
            opponent_deck.clear()
            break
        history.add(state)

        # Draw cards and determine whether to recurse
        pc = player_deck.popleft()
        oc = opponent_deck.popleft()
        recurse = with_recursion and len(player_deck) >= pc and len(opponent_deck) >= oc

        if recurse:
            # This round is won by whoever wins the subgame
            subgame_score = play_game(
                tuple(player_deck)[:pc],
                tuple(opponent_deck)[:oc],
                with_recursion=with_recursion
            )
            player_wins = subgame_score > 0
        else:
            # If not recursing, the highest card wins
            player_wins = pc > oc

        # Winner gets both card to the bottom of their deck (own card first)
        if player_wins:
            player_deck.extend((pc, oc))
        else:
            opponent_deck.extend((oc, pc))
        #        
    
    # Compute and return score
    res = compute_score(*player_deck) if player_deck else -1*compute_score(*opponent_deck)
    return res


def solve(data: str) -> tuple[int|str, ...]:
    player, opponent = parse(data)
    
    outcome = play_game(player, opponent)
    star1 = abs(outcome)
    print(f"Solution to part 1: {star1}")

    outcome2 = play_game(player, opponent, with_recursion=True)
    star2 = abs(outcome2)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 22
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
