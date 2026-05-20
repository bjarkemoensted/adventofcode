# `·`*  ·   · * · ` ·   *  .   · *·  `.  `·  `  ··*.· ` +  `· *·* `  · ·*    ·.`
# ·  ·`.+ ·*`·· `+.      `·  .·*   Category Six   · `+·*. ·+  ·.· * · `*· .··`*`
#  ·*`·  · .*` ·  · *· https://adventofcode.com/2019/day/23 *   `· ·.*   ·`.+.··
# `*··.*` ··  `    · *`  ·.·   `*`  ·     `· *· .  •   · *`·     +. ·`·` ·* ` ..

from aoc.aoc_2019.intcode import Computer


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


class Network:
    NAT = 255
    IDLE_TARGET = 0

    def __init__(self, program: list[int], n_addresses: int) -> None:
        self.computers = [Computer(program).add_input(i) for i in range(n_addresses)]
        self.nat_pending = (-1, -1)
        self.nat_hist: list[int] = []

    def _trigger_nat(self) -> None:
        """Callback for when the NAT detects an idle network.
        Sends the last package received by the NAT to the idle target computer."""
        x, y = self.nat_pending
        if x == -1 or y == -1:
            raise ValueError
        self.computers[self.IDLE_TARGET].add_input(x, y)
        self.nat_hist.append(y)

    def send_packet(self, adr: int, x: int, y: int) -> None:
        """Send a packet of (x, y) to the target address"""
        if adr == self.NAT:
            self.nat_pending = (x, y)
        else:
            self.computers[adr].add_input(x, y)
        #

    def run(self):
        """Runs a single pass over the network, allowing all machines to process
        their input queues and send any resulting packages."""

        idle = True
        for c in self.computers:
            # Check input queue
            if not c.stdin:
                c.add_input(-1)
            else:
                idle = False
            
            # Run and send any resulting packets
            c.run()
            while c.stdout:
                adr, x, y = c.read_stdout(n=3)
                self.send_packet(adr, x, y)
            #
        if idle:
            self._trigger_nat()
        
        return self
    
    def run_until_repeated_nat(self):
        """Keeps running the network until the NAT processes the same value twice in a row"""
        while len(self.nat_hist) < 2 or self.nat_hist[-1] != self.nat_hist[-2]:
            self.run()
        return self
        

def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)
    network = Network(program, n_addresses=50)
    nat_values = network.run_until_repeated_nat().nat_hist

    star1 = nat_values[0]
    print(f"Solution to part 1: {star1}")

    star2 = nat_values[-1]
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
