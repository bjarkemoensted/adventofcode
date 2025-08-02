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


test = """Immune System:
17 units each with 5390 hit points (weak to radiation, bludgeoning) with an attack that does 4507 fire damage at initiative 2
989 units each with 1274 hit points (immune to fire; weak to bludgeoning, slashing) with an attack that does 25 slashing damage at initiative 3

Infection:
801 units each with 4706 hit points (weak to radiation) with an attack that does 116 bludgeoning damage at initiative 1
4485 units each with 2961 hit points (immune to radiation; weak to fire, cold) with an attack that does 12 slashing damage at initiative 4"""

#2255 units each with 7442 hit points (weak to radiation) with an attack that does 31 bludgeoning damage at initiative 8


_pattern_raw = r"""
(?P<n_units>\d+) units.*?(?P<hit_points>\d+) hit points.
*?\(?(?P<resistance_stuff>.*?)\)?.*?
(?P<damage>\d+) (?P<damage_type>.*?) damage.*?
initiative (?P<initiative>\d+)
""".replace("\n", "")

_pattern = re.compile(_pattern_raw)


@dataclass
class Group:
    id_: uuid.UUID = field(default_factory=uuid.uuid4, init=False, repr=False)
    n_units: int
    hit_points: int
    damage: int
    damage_type: str
    initiative: int
    weak: tuple[_damage_type] = ()
    immune: tuple[_damage_type] = ()
    ind: int = -1
    army: str|None = None
    verbose: bool = field(default=False, repr=False)
    

    def estimate_damage(self, damage: int, type_: _damage_type):
        if type_ in self.weak:
            return 2*damage
        elif type_ in self.immune:
            return 0
        else:
            return damage
        #
    
    def __hash__(self) -> uuid.UUID:
        return hash(self.id_)

    @property
    def is_alive(self) -> bool:
        return self.n_units > 0

    @property
    def effective_power(self) -> int:
        return self.n_units * self.damage
    
    @property
    def short(self) -> str:
        return f"{self.army} group {self.ind}"
    
    def copy(self) -> Group:
        res = deepcopy(self)
        res.id_ = uuid.uuid4()
        return res

    def priority_target_selection(self) -> tuple[int, ...]:
        return (self.effective_power, self.initiative)

    def priority_enemy_selection(self, enemy: Group) -> tuple[int, ...]:
        damage = enemy.estimate_damage(damage=self.effective_power, type_=self.damage_type)
        res = (damage, enemy.effective_power, enemy.initiative)
        return res
    
    def priority_attack(self):
        return self.initiative

    def choose_victim(self, *enemies: Group) -> Group|None:
        candidates = (e for e in enemies if e.is_alive)
        res = max(candidates, key=self.priority_enemy_selection, default=None)
        return res
    
    def attack(self, other: Group):
        if not self.is_alive or not other.is_alive:
            return
        
        n_dead = other.receive_damage(damage=self.effective_power, type_=self.damage_type)
        if self.verbose:
            print(f"{self.short} attacks defending group {other.ind}, killing {n_dead} unit{'s'*int(n_dead != 1)}")

    def receive_damage(self, damage: int, type_: _damage_type) -> int:
        dmg = self.estimate_damage(damage=damage, type_=type_)
        n_dead = min(self.n_units, dmg // self.hit_points)
        self.n_units -= n_dead
        return n_dead
        

def _parse_group(s: str) -> dict[str, Any]:
    match = _pattern.search(s)
    if match is None:
        raise RuntimeError(f"No match - {s}")
    
    d = match.groupdict()
    
    kws = dict()

    resistance_parts = d.pop("resistance_stuff").split("; ")
    
    for part in resistance_parts:
        if not part:
            continue
        type_, vals_str = part.split(" to ")
        vals = tuple(vals_str.split(", "))
        kws[type_] = vals
    
    for k, v in d.items():
        if k != "damage_type":
            v = int(v)
        kws[k] = v

    return kws


def parse(s):
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
    def __init__(self, *groups: Group, verbose: bool = False):
        self._verbose = verbose
        self.parties: dict[str, list[Group]] = dict()

        for group in groups:
            self._add_group(group) 
        
        _parties = sorted(self.parties.keys())
        assert len(_parties) == 2
        self._enemy = {p1: _parties[(i+1) % len(_parties)] for i, p1 in enumerate(_parties)}
    
    @property
    def verbose(self) -> bool:
        return self._verbose
    
    @verbose.setter
    def verbose(self, val: bool):
        self._verbose = val
        for g in self._groups_order():
            g.verbose = val

    def _add_group(self, group: Group):
        army = group.army
        if army not in self.parties:
            self.parties[army] = []
        
        g = group.copy()
        g.verbose = self.verbose
        self.parties[army].append(g)

    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _status(self):
        for army, groups, in sorted(self.parties.items()):
            self._vprint(f"{army}:")
            for group in groups:
                self._vprint(f"Group {group.ind} contains {group.n_units} unit{'s'*(group.n_units != 1)}")

    def target_selection(self) -> dict[Group, Group]:
        targeted: set[Group] = set()

        res: dict[Group, Group] = dict()

        for army, groups in sorted(self.parties.items(), reverse=True):
            ordered = sorted(groups, key=lambda g: g.priority_target_selection(), reverse=True)
            for meh in ordered:
                other_army = self._enemy[army]
                
                cands = self.parties[other_army]
                wat = [c for c in cands if c not in targeted]
                target = meh.choose_victim(*wat)
                for other in wat:
                    self._vprint(f"{meh.short} would deal defending group {other.ind} {other.estimate_damage(meh.effective_power, meh.damage_type)} (chose this: {other is target})")
                
                if target is not None:
                    targeted.add(target)
                    res[meh] = target
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
        for army, groups in self.parties.items():
            
            self.parties[army] = [g for g in groups if g.is_alive]

    def is_active(self) -> bool:
        return all(any(elem.is_alive for elem in groups) for groups in self.parties.values())

    def tick(self):
        self._status()
        self._vprint()

        targets = self.target_selection()
        
        self.handle_attacks(targets)
        self.cleanup()
        self._vprint()

    def go(self):
        while self.is_active():
            self.tick()
        
        for v in self.parties.values():
            if v:
                return sum(g.n_units for g in v)


def solve(data: str):
    parsed = parse(data)
    battle = Battle(*parsed, verbose=False)
    
    battle.tick()

    #battle.verbose = True
    battle.tick()

    n_left = battle.go()

    # 12627 too high
    star1 = n_left
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2018, 24

    from aocd import get_data
    raw = get_data(year=year, day=day)
    #raw = test
    solve(raw)


if __name__ == '__main__':
    main()
