# .··`*+ `· *.` ·  .·• ·*   * ·`• ·.*. * · .`*    · · . `•  · *·. *· . · `·+·`.·
# ·. ·`· +`• *     ·*.  ·  `·   ·  Wait For It · ·* .* `.·   · .·*`*      .·*.·`
#  *.·  · *`·  . ·*``  https://adventofcode.com/2023/day/6 *+.· ` ·   * · +.·*`·
# · ·*.   · `+*· .·* · `.  · *  `     · . ·` *.  ·` * `*·  ·     ·.`·  ` ·*  · `

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class Race:
    duration: int
    record: int
    
    def distance_travelled(self, n_acc: int) -> int:
        """Returns the distance travelled when the accelerator is held for n_acc time steps"""

        if not 0 <= n_acc <= self.duration:
            raise ValueError("Invalid value. Can't accelerate for longer than race duration")
        
        speed = n_acc
        dist = speed*(self.duration - n_acc)
        return dist
    
    def ways_to_win_brute(self) -> int:
        """Brute forces the number of ways to win, by trying every possible initial
        acceleration steps."""
        res = 0
        for n_acc in range(self.duration + 1):
            dist = self.distance_travelled(n_acc=n_acc)
            beats_record = dist > self.record
            res += beats_record
        
        return res
    
    def ways_to_win(self) -> int:
        """Computes the number of ways to break the record by beaing sneaky, i.e.
        solving analytically for the records and computing the number of acceleration
        time steps in between."""
        
        record_low, record_high = self.solve_for_record()
        # Determine the first and last number of acceleration steps which exceeds the record
        a = math.floor(record_low + 1)
        b = math.ceil(record_high-1)
        res = b - a + 1
        return res
    
    def solve_for_record(self) -> tuple[float, float]:
        """Looks for the number of acceleration time steps required to hit the current
        record for the race. If we call the time steps t, and the race duration T, the total
        distance travelled is t*(T - t).
        Therefore, the current record can be found by solving the second degree polynomial
        -t**2 + t*T - record = 0"""
        
        a = -1
        b = self.duration
        c = -self.record
        
        d = b**2 - 4*a*c
        if d < 0:
            raise ValueError(f"Race {self} has unattainable record")
        
        s1, s2 = ((-b + factor*(d**0.5))/2*a for factor in (+1, -1))
        
        return s1, s2
    
    @classmethod
    def concatenate(cls, *races: Race) -> Race:
        """Combines races into a single race by string-concatenating their attributes"""
        duration_str = "".join(map(str, (race.duration for race in races)))
        record_str = "".join(map(str, (race.record for race in races)))
        res = cls(
            duration=int(duration_str),
            record=int(record_str)
        )
        
        return res

def parse(s: str) -> tuple[Race, ...]:
    """Parses into a tuple of race instances"""
    
    numbers = (tuple(int(elem) for elem in line.split(":")[1].strip().split()) for line in s.split("\n"))
    res = tuple(Race(duration=t, record=d) for t, d in zip(*numbers, strict=True))
    return res


def comb(*races: Race) -> int:
    res = 1
    for race in races:
        n_improvements_race = race.ways_to_win()
        res *= n_improvements_race
    
    return res


def solve(data: str) -> tuple[int|str, ...]:
    races = parse(data)
    
    star1 = comb(*races)
    print(f"Solution to part 1: {star1}")
    big_race = Race.concatenate(*races)
    
    star2 = comb(big_race)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 6
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
