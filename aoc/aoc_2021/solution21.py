# `* .·.·     ·+*· ·`   ·.  *  ·.·`+ .* ·.`·*    ·.`·   ·  *  ·`*·.  .*· + ·`.`·
#  .*· `   ··.`   ·*·  .  ` · . +   Dirac Dice  · ·* ·    *.·    +*. ·`  · ` ·*·
# `·. . `·· *   .  · . https://adventofcode.com/2021/day/21 .+ +·`· * ·.`.    ·.
# ·`· +·*`.    ·  .` ·   ·  .    `· *·.    · `·. *.·  ·  .`·*.·· .`     ·`·.· + 

from collections import defaultdict
from itertools import product
from typing import Iterable, Iterator, TypeAlias

# Alias for counts of the times a player has been in a given state
conftype: TypeAlias = tuple[tuple[int, int], tuple[int, int]]


def parse(s: str) -> tuple[int, int]:
    a, b = (int(line.split("position: ")[-1]) for line in s.splitlines())
    return a, b


def multiroll(rolls: Iterable[int], n_throws=3) -> Iterator[list[int]]:
    throws = []
    while True:
        for roll in rolls:
            throws.append(roll)
            if len(throws) >= n_throws:
                yield throws
                throws = []
            #
        #
    #


def play_game(rolls: Iterable[int], player_positions: Iterable[int], n_throws=3):
    player_positions = [val for val in player_positions]
    n = 0
    player_turn = -1
    points = [0 for _ in player_positions]
    cumrolls = multiroll(rolls, n_throws)

    while all(point < 1000 for point in points):
        player_turn = (player_turn + 1) % len(player_positions)
        roll = next(cumrolls)
        roll_sum = sum(roll)
        n += n_throws
        new_pos = 1 + ((player_positions[player_turn] + roll_sum - 1) % 10)
        player_positions[player_turn] = new_pos
        points[player_turn] += new_pos

    loser_points = min(points)
    num = loser_points*n
    return num


def get_roll_multiplicities(values: list[int], n_throws: int) -> dict[int, int]:
    res: dict[int, int] = {}
    for rolls in product(values, repeat=n_throws):
        s = sum(rolls)
        res[s] = res.get(s, 0) + 1
    return res


class QuantumGame:
    def __init__(self, player0pos: int, player1pos: int, win_at: int, die_values: list[int], n_throws: int) -> None:
        self.win_at = win_at
        self.next_player = 0

        # We count the positions and points of players like ((player0pos, player1points), ...): N
        self.state: dict[conftype|int, int] = defaultdict(int)
        # We start with a single universe with players in their starting positions and zero points
        player0 = (player0pos, 0)
        player1 = (player1pos, 0)
        self.state[(player0, player1)] += 1
        self.die = get_roll_multiplicities(values=die_values, n_throws=n_throws)

    def _find_next_configuration(self, configuration: conftype, player: int, roll: int) -> conftype|None:
        """For a given configuration ((player1pos, player1points), ...), current player, and roll, returns the resulting
        configuration. Returns None if the roll ends the game."""

        position, points = configuration[player]
        new_position = 1 + (position - 1 + roll) % 10
        new_points = points + new_position
        if new_points >= self.win_at:
            return None
        else:
            new_configuration = list(configuration)
            new_configuration[player] = (new_position, new_points)
            a, b = new_configuration
            return a, b

    def tick(self) -> bool:
        """Updates the game state."""
        player = self.next_player

        active_games = sum(v for k, v in self.state.items() if isinstance(k, tuple))
        done_games = sum(v for k, v in self.state.items() if isinstance(k, int))
        expected_games_after = done_games + 27 * active_games

        # Determine the new game states in all universes
        new_state: dict[conftype|int, int] = defaultdict(int)
        for configuration, multiplicity in self.state.items():
            if isinstance(configuration, int):
                new_state[configuration] += multiplicity
                continue
            for roll, roll_multiplicity in self.die.items():
                n_new_universes = multiplicity*roll_multiplicity
                new_configuration = self._find_next_configuration(configuration, player, roll)
                # If player N wins here, add to state[N]
                if new_configuration is None:
                    new_state[player] += n_new_universes
                else:
                    new_state[new_configuration] += n_new_universes
                #
            #

        games_after = sum(new_state.values())
        if games_after != expected_games_after:
            raise RuntimeError

        # Update state and next player
        done = new_state == self.state
        self.state = new_state
        self.next_player = (player + 1) % 2
        return done

    def play_game(self, max_rounds=None) -> None:
        if max_rounds is None:
            max_rounds = float("inf")
        round_ = 0
        done = False
        while not done:
            all_games_over = self.tick()
            round_ += 1
            done = all_games_over or round_ >= max_rounds

    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in sorted(self.state.items(), key=str) if v > 0)
    #


def solve(data: str) -> tuple[int|str, ...]:
    pos_player, pos_opponent = parse(data)
    star1 = play_game(
        rolls=range(1, 101),
        player_positions=(pos_player, pos_opponent)
    )
    print(f"Solution to part 1: {star1}")

    game = QuantumGame(pos_player, pos_opponent, win_at=21, die_values = [1,2,3], n_throws=3)
    game.play_game()
    star2 = max(v for k, v in game.state.items() if isinstance(k, int))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 21
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
