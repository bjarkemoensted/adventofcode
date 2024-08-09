import abc
from copy import deepcopy


def parse(s):
    raw = {}
    for line in s.split("\n"):
        k, v = line.split(": ")
        raw[k] = int(v)

    mapper = {"Hit Points": "hit_points", "Damage": "damage"}
    res = {mapper[k]: v for k, v in raw.items()}

    return res


class Character:
    def __init__(self, name, hit_points, mana=0, armor=0, damage=0):
        self.__name__ = name
        self.hit_points = hit_points
        self.mana = mana
        self.armor = armor
        self.damage = damage

    def __str__(self):
        s = f"{self.__name__} with {self.hit_points} Hit points, {self.mana} mana, and {self.armor} armor."
        return s


class InvalidActionException(Exception):
    pass


class Spell(metaclass=abc.ABCMeta):
    _cost = None

    def __init__(self, caster, target=None, verbose=False, register_mana_callback=None):
        self.caster = caster
        self.target = target
        self.verbose = verbose
        self.register_mana_callback = register_mana_callback

    @property
    def __name__(self):
        return str(self.__class__.__name__)

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def pay_mana(self):
        self.caster.mana -= self._cost
        if self.register_mana_callback is not None:
            self.register_mana_callback(self._cost)

    def check_castable(self):
        return self._cost <= self.caster.mana

    @abc.abstractmethod
    def casting_effect(self):
        pass

    def cast(self):
        if not self.check_castable():
            raise InvalidActionException("Cannot cast spell")
        self.pay_mana()
        self.casting_effect()


class Effect(Spell):
    _turns_active = None
    _timer = 0

    def check_castable(self):
        return self._timer == 0 and super().check_castable()

    def casting_effect(self):
        self._timer = self._turns_active
        self.effect_start()

    def effect_start(self):
        pass

    def effect_end(self):
        pass

    def apply_effect(self):
        pass

    def tick(self):
        """Apply any ongoing effects from spell"""
        if self._timer <= 0:
            return

        self.apply_effect()
        self._timer -= 1
        self._print(f"{self.__name__}'s timer is now {self._timer}.")
        # Apply effect that occurs when the effect spell ends
        if self._timer == 0:
            self.effect_end()


class MagicMissile(Spell):
    _cost = 53

    def casting_effect(self):
        self.target.hit_points -= 4


class Drain(Spell):
    _cost = 73

    def casting_effect(self):
        self.target.hit_points -= 2
        self.caster.hit_points += 2


class Shield(Effect):
    _cost = 113
    _turns_active = 6
    _boost = 7

    def effect_start(self):
        self.caster.armor += self._boost
        self._print(f"{self.caster.__name__} armor boosted to {self.caster.armor}")

    def effect_end(self):
        self.caster.armor -= self._boost
        self._print(f"{self.caster.__name__} armor reduced back to {self.caster.armor}")


class Poison(Effect):
    _cost = 173
    _turns_active = 6
    _damage = 3

    def apply_effect(self):
        self._print(f"{self.__name__} deals {self._damage} damage.")
        self.target.hit_points -= self._damage


class Recharge(Effect):
    _cost = 229
    _turns_active = 5
    _boost = 101

    def apply_effect(self):
        self._print(f"{self.__name__} provides {self._boost} mana.")
        self.caster.mana += self._boost


spellbook = [MagicMissile, Drain, Shield, Poison, Recharge]


class Game:
    def __init__(self, player, boss, verbose=False):
        self.player = deepcopy(player)
        self.boss = deepcopy(boss)
        self.verbose = verbose
        self.history = []
        self.total_mana_spent = 0
        self.spells = {}
        for constructor in spellbook:
            spell = constructor(
                caster=self.player,
                target=self.boss,
                verbose=verbose,
                register_mana_callback=self.add_mana
            )
            self.spells[spell.__name__] = spell

    def add_mana(self, mana):
        self.total_mana_spent += mana

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
        #

    def get_available_spells(self):
        """Returns a list of the spells which may be cast"""
        return list(self.spells.keys())

    def _print_status(self):
        s = f"- Player has {self.player.hit_points} hit points, {self.player.armor} armor and {self.player.mana} mana."
        s += f"\n- Boss has {self.boss.hit_points} hit points."
        self._print(s)

    def _tick_player(self, spell):
        self._print(f"-- Player turn --")
        self._print_status()

        self._apply_effects()
        if self.outcome:
            return

        self._print(f"Player casts {spell}.")
        self.spells[spell].cast()
        self._print()

    def _tick_boss(self):
        self._print(f"-- Boss turn --")
        self._print_status()

        self._apply_effects()
        if self.outcome:
            return

        damage = max(1, self.boss.damage - self.player.armor)
        self.player.hit_points -= damage
        self._print(f"-Boss attacks for {damage} damage.")
        self._print()

    def _apply_effects(self):
        for spell_name, spell in self.spells.items():
            if isinstance(spell, Effect):
                spell.tick()
            #
        #

    @property
    def outcome(self):
        if self.player.hit_points <= 0:
            return "loss"
        elif self.boss.hit_points <= 0:
            return "win"
        else:
            return None

    def tick(self, spell):
        self.history.append(spell)
        self._tick_player(spell)
        if self.outcome:
            self._print(f"*** {self.outcome.upper()} ***")
            return self.outcome

        self._tick_boss()
        if self.outcome:
            self._print(f"*** {self.outcome.upper()} ***")
            return self.outcome


def brute_force(starting_game: Game):
    best = float('inf')
    best_game = None
    longest_string = 0

    games = [deepcopy(starting_game)]
    n_its = 0
    while games:
        next_iteration = []
        for game in games:
            for spell in game.get_available_spells():
                new_game = deepcopy(game)
                try:
                    outcome = new_game.tick(spell=spell)
                except InvalidActionException:
                    continue
                # If the game isn't over, and we haven't exceeded the best (lowest) amount of mana, keep iterating game
                if outcome is None:
                    if new_game.total_mana_spent < best:
                        next_iteration.append(new_game)
                elif outcome == "win":
                    if new_game.total_mana_spent < best:
                        best = new_game.total_mana_spent
                        best_game = new_game
                    #
                #
            #
        n_its += 1
        games = next_iteration
        msg = f"Completed {n_its} growth iterations. Iterating {len(games)} games in next cycle. Mana record: {best}."
        longest_string = max(longest_string, len(msg))
        msg += (longest_string - len(msg))*" "
        print(msg, end="\r")
    print()

    return best_game


class HardGame(Game):
    def _tick_player(self, spell):
        self.player.hit_points -= 1
        if self.outcome:
            return
        return super()._tick_player(spell)
    #


def solve(data: str):
    boss_stats = parse(data)
    player_stats = {'hit_points': 50, "mana": 500}

    player = Character("Player", **player_stats)
    boss = Character("Boss", **boss_stats)

    game1_initial = Game(player, boss)
    game1_final = brute_force(starting_game=game1_initial)
    star1 = game1_final.total_mana_spent
    print(f"Solution to part 1: {star1}")

    hard_game_initial = HardGame(player, boss)
    hard_game_final = brute_force(hard_game_initial)
    star2 = hard_game_final.total_mana_spent
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2015, 22
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
