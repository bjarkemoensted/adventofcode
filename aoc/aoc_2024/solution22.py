# ·` · . ·  • `.·  + · .`  *··.•. * `·    ·*`.· + ·· `*.` ·     +`·   · *   `*··
# `·*`  . · ·  *` +·.   *·      · Monkey Market  ` * · `·   *  `.·` ·  *·`+ .· *
# · .  +·  `.·*·+   `. https://adventofcode.com/2024/day/22 .·    +·   ` ·.·* .`
# . ·*·`      ·`  · *`+ ·. ·     ·`    .·  +*· `·  ·`.·+     * ·  .* ·`   ·.·`·•


from collections import Counter
import math
import numba


def parse(s: str):
    res = [int(line) for line in s.splitlines()]
    return res


@numba.njit()
def next_secret(secret: int):
    """Takes a 'secret' integer and returns the next pseudorandom secret."""
    res = secret^(secret * 64) 
    res %= 16777216
    res = res^(res // 32)
    res %= 16777216
    res = res^(res * 2048)
    res %= 16777216
    
    return res


def crunch(seeds, n_generations=2000, n_hist=4):
    """Takes the random seeds from the riddle input, and performs the specified number of iterations for each seed.
    This results in the first N pseudorandom number for each monkey.
    For each monkey, we store a dict mapping historcal price changes (of the specified number of time steps), to the
    price of bananas should the broker monkey make a bid at the first occurrence of that sequence.
    
    Returns: list of final secrets, dict mapping sequences to total prices (summed over all monkeys)"""
    
    final_secrets = []
    totals_by_history = Counter()
    
    # Determine the number of bits needed to represent history in an integer
    max_change = 9
    n_bits_needed = math.ceil(math.log2(2*max_change)) 
    mod = 2**(n_hist*n_bits_needed)
    
    for seed in seeds:
        d = dict()  # Maps historical price changes to prices for this monkey
        
        # Determine the initial secret, price, and history for this monkey
        secret = seed
        price = secret % 10
        hist = 0
        
        for i in range(n_generations):
            # Update secret and price data
            secret = next_secret(secret)
            new_price = secret % 10
            change = new_price - price
            price = new_price
            
            # Update history - shift n bits left, then take module to keep only 4 most recent changes
            hist = (hist << n_bits_needed) + (change + max_change)
            hist %= mod
            
            # Only add to mapping if history has sufficient length and occurs for the first time
            if i+1 < n_hist or hist in d:
                continue
            
            d[hist] = price

        # Update final results
        final_secrets.append(secret)
        for k, v in d.items():
            totals_by_history[k] += v
        
    
    return final_secrets, totals_by_history


def solve(data: str) -> tuple[int|str, int|str]:
    seeds = parse(data)
    final_secrets, totals_by_history = crunch(seeds=seeds)
    
    star1 = sum(final_secrets)
    print(f"Solution to part 1: {star1}")

    star2 = max(totals_by_history.values())
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 22
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()