#  *ꞏ..•⸳* `  ꞏ⸳ .`   ꞏ*  ꞏ⸳.   ꞏ . `⸳  .ꞏꞏ⸳*    ⸳ *ꞏ.     ⸳ꞏ+  `*.   `*  ꞏ ` ꞏ.
# * ꞏ ⸳  ꞏ+.` .  *  ⸳` ꞏ*  Immune System Simulator 20XX      ꞏ *⸳ ꞏ `      ⸳⸳ . 
#    ⸳ꞏ ꞏ  •⸳.`  ꞏ   ꞏ https://adventofcode.com/2018/day/24  ꞏ ⸳•. ⸳•  .ꞏ    ꞏ•.
# ꞏ. `+⸳ .ꞏ⸳ + ꞏ`  ꞏ*.⸳  ꞏ` *  .   ꞏ   ⸳ ꞏ  * .ꞏ*`  .`* +⸳`⸳  ꞏ .ꞏ * ` ꞏ.⸳. * ꞏ⸳

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any, Literal, TypeAlias
import uuid


_damage_type: TypeAlias = Literal["radiation", "bludgeoning", "fire", "cold", "slashing"]
_army_type: TypeAlias = Literal["Immune System", "Infection"]


test = """Immune System:
17 units each with 5390 hit points (weak to radiation, bludgeoning) with an attack that does 4507 fire damage at initiative 2
989 units each with 1274 hit points (immune to fire; weak to bludgeoning, slashing) with an attack that does 25 slashing damage at initiative 3

Infection:
801 units each with 4706 hit points (weak to radiation) with an attack that does 116 bludgeoning damage at initiative 1
4485 units each with 2961 hit points (immune to radiation; weak to fire, cold) with an attack that does 12 slashing damage at initiative 4"""

_pattern_raw = r"""
(?P<n_units>\d+) units.*?(?P<hit_points>\d+) hit points.
.*?\(?(?P<resistance_stuff>[^)]*)\)?.*?
with an attack that does (?P<damage>\d+) (?P<damage_type>.*?) damage.*?
initiative (?P<initiative>\d+)
""".replace("\n", "")

_pattern = re.compile(_pattern_raw)


def _parse_group(s: str) -> dict[str, Any]:
    """Parses a line describing a group into a dict of kwargs for initializing a Group dataclass."""
    
    match = _pattern.search(s)
    if match is None:
        raise RuntimeError(f"No match - {s}")
    
    kws: dict[str, Any] = dict()
    d = match.groupdict()

    # Parse the resistance + vulnerability info in parentheses (if present)
    resistance_parts = d.pop("resistance_stuff").split("; ")
    for part in resistance_parts:
        if not part:
            continue
        # map weakness and immunity to a tuple of the damage types to which the unit is weak/immune
        type_, vals_str = part.split(" to ")
        vals = tuple(vals_str.split(", "))
        kws[type_] = vals
    
    # Convert quantities into ints
    for k, v in d.items():
        val = v
        if k != "damage_type":
            val = int(v)
        kws[k] = val

    return kws


@dataclass
class Group:
    """Represents a group of units in the reindeer immune system vs infection showdown"""

    id_: uuid.UUID = field(default_factory=uuid.uuid4, init=False, repr=False)
    army: _army_type
    ind: int
    n_units: int
    hit_points: int
    damage: int
    damage_type: _damage_type
    initiative: int
    weak: tuple[_damage_type, ...] = ()
    immune: tuple[_damage_type, ...] = ()
    verbose: bool = field(default=False, repr=False)
    
    def estimate_damage(self, damage: int, type_: _damage_type):
        """Estimates damage to this unit given the specified amount of raw damage, and the damage type."""
        if type_ in self.weak:
            return 2*damage
        elif type_ in self.immune:
            return 0
        else:
            return damage
        #
    
    def __hash__(self):
        return hash(self.id_)

    @property
    def is_alive(self) -> bool:
        return self.n_units > 0

    @property
    def effective_power(self) -> int:
        res = self.n_units * self.damage
        return res
    
    @property
    def short(self) -> str:
        return f"{self.army} group {self.ind}"
    
    def copy(self) -> Group:
        """Take a copy of the group. Assigns a new uuid to avoid mistakes when hashing."""
        res = deepcopy(self)
        res.id_ = uuid.uuid4()
        return res

    def priority_target_selection(self) -> tuple[int, ...]:
        """Sorting key function for determining order of target selection"""
        return (self.effective_power, self.initiative)

    def priority_enemy_selection(self, enemy: Group) -> tuple[int, ...]:
        """Sorting key for choosing 'best' enemy group to attack"""
        damage = enemy.estimate_damage(damage=self.effective_power, type_=self.damage_type)
        res = (damage, enemy.effective_power, enemy.initiative)
        return res
    
    def priority_attack(self):
        return self.initiative

    def choose_victim(self, *enemies: Group) -> Group|None:
        """Chooses an enemy to attack, from the list of candidates.
        If no enemy can be targeted, returns None."""
        candidates = (e for e in enemies if e.is_alive and e.estimate_damage(self.effective_power, type_=self.damage_type))
        res = max(candidates, key=self.priority_enemy_selection, default=None)
        return res
    
    def attack(self, other: Group):
        # Abort attack if either unit has died
        if not self.is_alive or not other.is_alive:
            return
        
        # Execute the attack
        n_dead = other.receive_damage(damage=self.effective_power, type_=self.damage_type)
        if self.verbose:
            print(f"{self.short} attacks defending group {other.ind}, killing {n_dead} unit{'s'*int(n_dead != 1)}")

    def receive_damage(self, damage: int, type_: _damage_type) -> int:
        """Take x damage of the specified type. Returns the number of casualties"""
        dmg = self.estimate_damage(damage=damage, type_=type_)
        n_dead = min(self.n_units, dmg // self.hit_points)
        self.n_units -= n_dead

        return n_dead


def parse(s) -> list[Group]:
    parts = s.split("\n\n")
    res = []
    
    for part in parts:
        lines = part.splitlines()
        army = lines[0].split(":")[0]

        for i, line in enumerate(lines[1:]):
            kws = _parse_group(line)
            ind = i + 1
            g = Group(**kws, ind=ind, army=army)
            res.append(g)
        #

    return res


class Battle:
    def __init__(self, *groups: Group, boost: int=0, verbose: bool = False):
        """Setup a battle with the specified units, and an optional boost for the immune system."""
        
        self._verbose = verbose
        self.parties: dict[_army_type, list[Group]] = dict()
        # Store copy of the groups so we can re-initialize at any time
        self._raw = deepcopy(groups)
        
        _parties = sorted({g.army for g in groups})
        assert len(_parties) == 2
        self._enemy = {p1: _parties[(i+1) % len(_parties)] for i, p1 in enumerate(_parties)}
        self.reset(boost=boost)
    
    def reset(self, boost: int=0):
        """Reset the battle, with the specified immune system boost (defaults to 0)"""
        
        self.parties = {k: [] for k in self._enemy.keys()}
        
        for group in self._raw:
            g = group.copy()
            g.verbose = self.verbose
            if g.army == "Immune System":
                g.damage += boost
            self.parties[g.army].append(g)
        #
    
    def signature(self) -> tuple[tuple[int, ...], ...]:
        """Returns a 'signature' tuple representing the current state of the battle, in terms
        of the number of units in each group for each side.
        This can be used to detect locked states, e.g. if remaining units don't have enough
        attack power to hurt the other."""

        res = tuple(
            tuple(g.n_units for g in sorted(groups, key=lambda g_: g_.ind))
            for _, groups in sorted(self.parties.items())
        )
        
        return res
    
    @property
    def verbose(self) -> bool:
        return self._verbose
    
    @verbose.setter
    def verbose(self, val: bool):
        """Propagate verbosity to all groups in the battle"""
        self._verbose = val
        for groups in self.parties.values():
            for g in groups:
                g.verbose = val
            #
        #

    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _status(self):
        """Print status if verbose (for troubleshooting)"""
        for army, groups, in sorted(self.parties.items()):
            self._vprint(f"{army}:")
            for group in groups:
                self._vprint(f"Group {group.ind} contains {group.n_units} unit{'s'*(group.n_units != 1)}")

    def target_selection(self) -> dict[Group, Group]:
        """Performs target selection according to the rules"""
        
        targeted: set[Group] = set()
        res: dict[Group, Group] = dict()

        for army, groups in sorted(self.parties.items(), reverse=True):
            # For each army, go over the units in their order of priority
            ordered = sorted(groups, key=lambda g: g.priority_target_selection(), reverse=True)
            for group in ordered:
                other_army = self._enemy[army]
                # Choose preferred victim from list of yet-untargeted enemies
                candidates = [c for c in self.parties[other_army] if c not in targeted]
                target = group.choose_victim(*candidates)
                if target is not None:
                    targeted.add(target)
                    res[group] = target
                
                # Print explanation if verbose
                for other in candidates:
                    dmg_est = other.estimate_damage(group.effective_power, group.damage_type)
                    self._vprint(f"{group.short} would deal defending group {other.ind} {dmg_est}")
                #
            #
        self._vprint()
        
        return res
    
    def handle_attacks(self, targets: dict[Group, Group]):
        for attacker in sorted(targets.keys(), key=lambda g: g.priority_attack(), reverse=True):
            
            target = targets[attacker]
            attacker.attack(target)
        #

    def cleanup(self):
        """Remove dead units from the battle"""
        for army, groups in self.parties.items():
            self.parties[army] = [g for g in groups if g.is_alive]

    def is_active(self) -> bool:
        return all(any(elem.is_alive for elem in groups) for groups in self.parties.values())

    def tick(self, n: int=1) -> bool:
        """Iterate a number of rounds of combat.
        Returns a bool indicating whether the fight is over. This includes
        situations where a side has one, or where the battle is locked (units are immune to each other's attacks).
        use n = -1 (or any negative number) to repeat until the battle is over"""
        
        self._status()
        self._vprint()
        sig = self.signature()
        
        while n != 0:
            # Do target selection. Terminate if no one is targeted
            targets = self.target_selection()
            if not targets:
                return False
            
            # Process the attacks and remove any dead groups
            self.handle_attacks(targets)
            self.cleanup()
            self._vprint()
            
            # Check if we're stuck
            n -= 1
            new_sig = self.signature()
            locked = new_sig == sig

            if locked or not self.is_active():
                return False
            
            sig = new_sig
            #
        
        return True
        
    def go(self) -> int|None:
        """Simulate the battle, returning the number of remaining units on the winning side.
        If the immune system is victorious, the number is negative, othersise, it's positive.
        If the battle becomes locked indefinitely, returns None"""
        
        self.tick(-1)
        
        # Find the army with any remaining groups
        for army, groups in self.parties.items():
            if not groups:
                continue
            
            # If the army still has enemies, the battle didn't eend
            still_has_enemies = self.parties[self._enemy[army]]
            if still_has_enemies:
                return None
            
            factor = 1 if army == "Infection" else -1
            res = factor*sum(g.n_units for g in groups)
            
            return res
            #
        
        raise RuntimeError("Battle finished with no units left on either side")


def compute_min_boost(battle: Battle) -> int:
    """Determines the minimum attack power required before the reindeer immune system wins.
    Repeatedly resets the battle and uses binary search to determine the least required boost."""
    
    low = 0
    high = 16
    outcomes: dict[int, int|None] = dict()
    
    def reindeer_makes_it(n: int) -> bool:
        """Determine whether the reindeer's immune system prevails"""
        
        nonlocal battle, outcomes
        battle.reset(boost=n)
        
        # Just record the outcome (if any). We can grab the correct one later
        outcome = battle.go()
        outcomes[n] = outcome
        if outcome is None:
            return False
        return outcome < 0
    
    # Keep increasing 'high' until we hit a point where the elves make it
    while not reindeer_makes_it(high):
        low = high
        high *= 2
    
    # Keep narrowing the range until we find the turning point
    while high - low > 1:
        mid = (high + low) // 2
        if reindeer_makes_it(mid):
            high = mid
        else:
            low = mid
        #
    
    min_boost = min(k for k, v in outcomes.items() if isinstance(v, int) and v < 0)
    outcome = outcomes[min_boost]
    assert isinstance(outcome, int)
    res = -outcome
    return res


def solve(data: str):
    parsed = parse(data)
    battle = Battle(*parsed, verbose=False)
    
    star1 = battle.go()
    print(f"Solution to part 1: {star1}")
    
    star2 = compute_min_boost(battle)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2018, 24
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
