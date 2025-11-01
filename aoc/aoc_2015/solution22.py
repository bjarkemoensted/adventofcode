# `. ·.*·. `·+ `·  *· ·  `   .·   ` *   +·`·.   ·*  ·· .      ··.  *· .  ··`  .·
# . ` ·   ·* ·.·  ·     .· `+ Wizard Simulator 20XX · `·  .• ·    ··. `··.* ·.· 
# ··•`*  ·    ·.    ·· https://adventofcode.com/2015/day/22 ·*`. ·`.  ·  · · `.·
# .·· `. •*·.`   · ·` ·* .·  `· •.· · `+. · ·  .·•`· `.  ··.*·   `  ··.*     ·`·


import heapq


def parse(s: str):
    raw = {}
    for line in s.split("\n"):
        k, v = line.split(": ")
        raw[k] = int(v)

    mapper = {"Hit Points": "boss_hp", "Damage": "boss_damage"}
    res = {mapper[k]: v for k, v in raw.items()}

    return res


class Queue:
    """Queue thingy for the A* algorithm"""
    def __init__(self):
        self._items = []
        self._set = set([])

    def push(self, item, priority):
        heapq.heappush(self._items, (priority, item))
        self._set.add(item)

    def pop(self):
        _, item = heapq.heappop(self._items)
        self._set.remove(item)
        return item

    def __bool__(self):
        return bool(self._items)

    def __len__(self):
        return len(self._items)

    def __contains__(self, item):
        return item in self._set
    #


class InvalidMoveError(Exception):
    pass


class Game:
    player_hp_ind = 0
    player_mana_ind = 1
    boss_hp_ind = 2
    shield_timer_ind = 3
    poison_timer_ind = 4
    recharge_timer_ind = 5

    shield_effect = 7
    recharge_effect = 101

    spells = ("magic_missile", "drain", "shield", "poison", "recharge")
    spell_costs = dict(
        magic_missile=53,
        drain=73,
        shield=113,
        poison=173,
        recharge=229
    )

    spell_damage = dict(
        magic_missile=4,
        drain=2,
        poison=3
    )

    spell_duration = dict(
            shield=6,
            poison=6,
            recharge=5
        )

    def __init__(
            self,
            boss_hp,
            boss_damage,
            player_hp=50,
            player_mana=500,
            hard_mode=False,
            verbose=False
        ):

        self.boss_damage = boss_damage
        self.boss_hp = boss_hp
        self.player_hp = player_hp
        self.player_mana = player_mana
        self.hard_mode = hard_mode
        self.verbose = verbose

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def make_initial_state(self) -> tuple[int, ...]:
        vals = [0 for _ in range(6)]
        vals[self.player_hp_ind] = self.player_hp
        vals[self.player_mana_ind] = self.player_mana
        vals[self.boss_hp_ind] = self.boss_hp
        res = tuple(vals)
        return res

    def magic_missile(self, state: list[int]) -> None:
        state[self.boss_hp_ind] -= self.spell_damage["magic_missile"]
        state[self.player_mana_ind] -= self.spell_costs["magic_missile"]

    def drain(self, state: list[int]) -> None:
        state[self.boss_hp_ind] -= self.spell_damage["drain"]
        state[self.player_hp_ind] += self.spell_damage["drain"]
        state[self.player_mana_ind] -= self.spell_costs["drain"]

    def poison(self, state: list[int]) -> None:
        if state[self.poison_timer_ind] != 0:
            raise InvalidMoveError
        state[self.player_mana_ind] -= self.spell_costs["poison"]
        state[self.poison_timer_ind] = self.spell_duration["poison"]

    def shield(self, state: list[int]) -> None:
        if state[self.shield_timer_ind] != 0:
            raise InvalidMoveError
        state[self.player_mana_ind] -= self.spell_costs["shield"]
        state[self.shield_timer_ind] = self.spell_duration["shield"]

    def recharge(self, state: list[int]) -> None:
        if state[self.recharge_timer_ind] != 0:
            raise InvalidMoveError
        state[self.player_mana_ind] -= self.spell_costs["recharge"]
        state[self.recharge_timer_ind] = self.spell_duration["recharge"]

    def game_over(self, state: list[int]|tuple[int, ...]) -> bool:
        """Returns whether the game is over (player or boss is dead)"""
        return any(state[ind] <= 0 for ind in (self.player_hp_ind, self.boss_hp_ind))

    def won(self, state: list[int]|tuple[int, ...]) -> bool:
        """Returns whether the game is won"""
        return state[self.boss_hp_ind] <= 0 and state[self.player_hp_ind] > 0

    def _apply_effects(self, state: list[int]) -> None:
        """Applies all effects currently in play, and decrements their timers"""

        if self.game_over(state):
            return

        if state[self.shield_timer_ind] > 0:
            # Shield doesn't really do anything when the timer ticks. Armor effect on damage is handled during attacks
            state[self.shield_timer_ind] -= 1
            self.vprint(f"Shield provides {self.shield_effect} Armor. Timer now at {state[self.shield_timer_ind]}.")
        if state[self.poison_timer_ind] > 0:
            # Handle poison damage and timer
            state[self.poison_timer_ind] -= 1
            dmg = self.spell_damage["poison"]
            state[self.boss_hp_ind] -= dmg
            self.vprint(f"Poison deals {dmg} damage. timer is now {state[self.poison_timer_ind]}")
        if state[self.recharge_timer_ind] > 0:
            # Handle mana recharge effects
            state[self.recharge_timer_ind] -= 1
            state[self.player_mana_ind] += self.recharge_effect
            msg = f"Recharge provides {self.recharge_effect} mana."
            msg += f"Its timer is now {state[self.recharge_timer_ind]}"
            self.vprint(f"Recharge gives {self.recharge_effect} mana. Timer now at {state[self.recharge_timer_ind]}.")
        #

    def boss_turn(self, state: list[int]) -> None:
        """Runs the boss turn - attack taking any shield effect into consideration"""

        if self.game_over(state):
            return
        
        self.vprint("-- Boss turn --")
        self.vprint(f"- State: {state}")

        dmg = self.boss_damage
        if state[self.shield_timer_ind] > 0:
            dmg = max(1, dmg - self.shield_effect)

        state[self.player_hp_ind] -= dmg
        self.vprint(f"Boss attacks for {dmg} damage.")
        self.vprint(f"State after turn: {state}")

    def player_turn(self, state: list[int], spell: str) -> None:
        """Runs the player turn"""

        if self.hard_mode:
            state[self.player_hp_ind] -= 1

        self.vprint("-- Player turn --")
        self.vprint(f"- State: {state}")
        self.vprint(f"Player casts {spell}")

        # Cast the selected spell
        fun = getattr(self, spell)
        fun(state)

        # Invalid move if we run out of mana
        if state[self.player_mana_ind] < 0:
            raise InvalidMoveError

        self.vprint(f"State after turn: {state}")

    def turn(self, state: tuple[int, ...], spell: str|list[str]) -> tuple[int, ...]:
        """Run one or multiple spell casts"""

        if isinstance(spell, list):
            running = state
            for spell_ in spell:
                running = self.turn(state=running, spell=spell_)
            return running

        new_state = [val for val in state]
        self.player_turn(state=new_state, spell=spell)
        self._apply_effects(state=new_state)
        self.vprint()
        self.boss_turn(state=new_state)
        self._apply_effects(state=new_state)
        self.vprint()

        res = tuple(new_state)
        return res

    def neighbors(self, state: tuple):
        """Get all states it's possible to get to from the input state"""

        res = []
        for spell in self.spells:
            cost = self.spell_costs[spell]
            # Don't even try the ones we can't afford anyway
            if cost > state[self.player_mana_ind]:
                continue
            try:
                # Just try to cast any remaining spell, and handle exceptions if not possible
                neighbor = self.turn(state=state, spell=spell)
            except InvalidMoveError:
                continue

            res.append((spell, cost, neighbor))
        return res

    def heuristic(self, state: tuple):
        """Could do somehthing smart here, but this runs plenty fast without an informative heuristic"""
        return 0


def play(game: Game):
    """A*, wizarding version."""

    initial_state = game.make_initial_state()

    d_g = dict()
    d_g[initial_state] = 0

    f_initial = game.heuristic(state=initial_state)

    open_ = Queue()
    open_.push(initial_state, priority=f_initial)

    last_spell: dict[str, tuple[int, ...]] = dict()  # Keep track of the spells cast
    camefrom: dict[tuple[int, ...], tuple[int, ...]] = dict()

    while open_:
        current = open_.pop()
        if game.won(current):
            # Reconstruct the history of cast spells
            link = current
            rev = []
            while link in camefrom:
                spell = last_spell[link]
                link = camefrom[link]
                rev.append(spell)
            res = rev[::-1]
            return res
        elif game.game_over(current):
            continue  # Don't continue if the game is lost

        for spell, cost, neighbor_state in game.neighbors(state=current):
            g_tentative = d_g[current] + cost
            improved = g_tentative < d_g.get(neighbor_state, float("inf"))
            if improved:
                last_spell[neighbor_state] = spell
                camefrom[neighbor_state] = current
                d_g[neighbor_state] = g_tentative
                h = game.heuristic(state=neighbor_state)
                f = g_tentative + h
                open_.push(neighbor_state, priority=f)
                #
            #
        #
    #


def solve(data: str) -> tuple[int|str, int|str]:
    boss_stats = parse(data)
    game = Game(**boss_stats)

    path = play(game=game)
    star1 = sum(game.spell_costs[spell] for spell in path)
    print(f"Solution to part 1: {star1}")

    hard_game = Game(**boss_stats, hard_mode=True)
    path2 = play(hard_game)

    star2 = sum(game.spell_costs[spell] for spell in path2)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 22
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
