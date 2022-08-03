from copy import deepcopy
import json

# Read in data
with open("input22.txt") as f:
    puzzle_input = f.read()


def parse(s):
    res = {}
    for line in s.split("\n"):
        k, v = line.split(": ")
        res[k] = int(v)

    return res


boss_stats = parse(puzzle_input)
player_stats = {'Hit Points': 50, "Mana": 500, "Armor": 0}


costs = {
    "Magic Missile": 53,
    "Drain": 73,
    "Shield": 113,
    "Poison": 173,
    "Recharge": 229
}


class InvalidAction(Exception):
    pass


class Game:
    all_actions = ["Recharge", "Shield", "Poison", "Magic Missile", "Drain"]

    def __init__(self, player, boss, verbose=False, hard=False):
        self.player = deepcopy(player)
        self.boss = deepcopy(boss)
        self.effects = {
            "Recharge": 0,
            "Shield": 0,
            "Poison": 0
        }
        self.verbose = verbose
        self.mana_spent = 0
        self.hard = hard

    def __str__(self):
        keys = ["boss", "player", "effects"]
        d = {k: getattr(self, k) for k in keys}
        s = json.dumps(d, sort_keys=True, indent=4)
        return s

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _action_allowed(self, action):
        if action in self.effects and self.effects[action] > 0:
            return False
        if costs[action] > self.player["Mana"]:
            return False
        return True

    def get_allowed_actions(self):
        actions = sorted([action for action in self.all_actions if self._action_allowed(action)])
        return actions

    def apply_effects(self):
        if self.effects["Shield"] > 0:
            self.effects["Shield"] -= 1
            if self.effects["Shield"] == 0:
                self.player["Armor"] = 0
            #
        if self.effects["Recharge"] > 0:
            self.player["Mana"] += 101
            self.effects["Recharge"] -= 1
        if self.effects["Poison"] > 0:
            self.boss["Hit Points"] -= 3
            self.effects["Poison"] -= 1

    def cast_spell(self, spell):
        if spell not in self.get_allowed_actions():
            raise InvalidAction

        self.player["Mana"] -= costs[spell]
        if self.player["Mana"] < 0:
            raise InvalidAction
        self.mana_spent += costs[spell]

        if spell == "Magic Missile":
            self.boss["Hit Points"] -= 4
        elif spell == "Drain":
            self.boss["Hit Points"] -= 2
            self.player["Hit Points"] += 2
        elif spell == "Shield":
            self.player["Armor"] = 7
            self.effects["Shield"] = 6
        elif spell == "Poison":
            self.effects["Poison"] = 6
        elif spell == "Recharge":
            self.effects["Recharge"] = 5

    def tick(self, action):
        # player turn
        self.print("-- Player turn --")
        self.print("- Player has {Hit Points} hit points, {Armor} armor, and {Mana} mana.".format(**self.player))
        self.print(f"- Boss has {self.boss['Hit Points']} hit points.")
        if self.hard:
            self.player["Hit Points"] -= 1
        if self.player["Hit Points"] <= 0:
            return "Loss"
        self.print(f"Player casts {action}.")
        self.print()

        self.apply_effects()
        if self.boss["Hit Points"] <= 0:
            self.print("Boom!")
            return "Win"
        self.cast_spell(action)

        # boss turn
        self.print("-- Boss turn --")
        self.print("- Player has {Hit Points} hit points, {Armor} armor, and {Mana} mana.".format(**self.player))
        self.print(f"- Boss has {self.boss['Hit Points']} hit points.")

        self.apply_effects()
        if self.boss["Hit Points"] <= 0:
            self.print("Boom!")
            return "Win"
        damage = max(1, self.boss["Damage"] - self.player["Armor"])
        self.player["Hit Points"] -= damage
        if self.player["Hit Points"] <= 0:
            return "Loss"

        attackstring = str(max(1, self.boss["Damage"] - self.player["Armor"]))
        if self.player["Armor"]:
            attackstring = f'{self.boss["Damage"]} - {self.player["Armor"]} = ' + attackstring

        self.print(f"Boss attacks for {attackstring} damage!")

        self.print()


def bfs(initial_game):
    lowest_spent = float("inf")
    games = [deepcopy(initial_game)]
    completed_games = []
    done = False
    n_its = 0

    while not done:
        # List of the games to 'grow' (applying all possible actions and noting the resulting new states)
        new_states = []
        for i in range(len(games)-1, -1, -1):
            game = games[i]

            allowed = game.get_allowed_actions()
            # If no actions are possible, the game is through
            if not allowed:
                del games[i]
                continue

            # Try every possible action
            for action in game.get_allowed_actions():
                newgame = deepcopy(game)
                outcome = newgame.tick(action)
                if outcome is None:
                    if newgame.mana_spent < lowest_spent:
                        new_states.append(newgame)
                else:
                    completed_games.append(newgame)
                    if outcome == "Win":
                        lowest_spent = min(lowest_spent, newgame.mana_spent)
                #
            #
        n_its += 1
        games = new_states
        done = len(games) == 0
        msg = f"Ran {n_its} iterations. Grew {len(games)} games states. Got {len(games)} games total."
        msg += f" Lowest boss health: {min([g.boss['Hit Points'] for g in games+completed_games])}."

        print(msg, end="\r")
    print()

    return lowest_spent


game1 = Game(player=player_stats, boss=boss_stats)
best = bfs(game1)
print(f"Lowest amount of mana spent on winning game: {best}.")

game2 = Game(player=player_stats, boss=boss_stats, hard=True)
best2 = bfs(game2)
print(f"Lowest amount of mana spent on winning a hard game: {best2}.")
# 1295 too high