# .`•· ··•* .    · ·`.  * ·  `   ..·+ ·`  •*` · ·. *·   •  `·.`  *·   *· ·.` •  
#  ·*`    .· *·   · . ·+       Allergen Assessment   ·` · .·  * ·    *·  .` ·.*·
# ·.   + ·· `.  ·  .*` https://adventofcode.com/2020/day/21 .  ·    *`· .+ ·`··.
#   ·*      ··`* .  ·       *·`·.  · *  .·· +*.    ·`.·   `.• ·* ·`·  • · · * .·


from dataclasses import dataclass


@dataclass
class Food:
    ingredients: tuple[str, ...]
    allergens: tuple[str, ...]


def parse(s: str) -> list[Food]:
    res = []
    for line in s.splitlines():
        ingpart, alpart = line[:-1].split(" (contains ")
        ingredients = tuple(ingpart.split())
        allergens = tuple(alpart.split(", "))
        res.append(Food(ingredients=ingredients, allergens=allergens))
    return res


def determine_allergens(*foods: Food) -> dict[str, str]:
    """Given the input foods, determine a mapping from allergens to ingredients.
    This works by maintaining a set of possible ingredients for each allergen,
    continually taking those with only one possibility and excluding from the
    remaining options."""
    
    # map allergens to possible ingredients
    unknown: dict[str, set[str]] = dict()

    for food in foods:
        _ingredients = set(food.ingredients)
        for al in food.allergens:
            unknown[al] = unknown.get(al, _ingredients).intersection(_ingredients)
    
    # Keep track of the ones that are definitely settled
    determined: dict[str, str] = dict()

    while unknown:
        # Look for allergens with only one possibly ingredients
        distinct = [(al, ing) for al, ing in unknown.items() if len(ing) == 1]
        if not distinct:
            raise RuntimeError
        
        # Add the now-distinct allergens to result
        remove = set()
        for al, ings in distinct:
            ing = unknown.pop(al).pop()
            determined[al] = ing
            remove.add(ing)
        
        # Remove from unknown
        for al, ings in unknown.items():
            unknown[al] = ings - remove
        #
    
    return determined


def solve(data: str) -> tuple[int|str, ...]:
    foods = parse(data)
    
    # Map allergens to ingredients
    known_allergens = determine_allergens(*foods)
    # Get all the ingredients where an allergen can be determined
    allergenic = set(known_allergens.values())
    # Count how many listed ingredients are absent from the known set
    star1 = sum(ing not in allergenic for food in foods for ing in food.ingredients)
    print(f"Solution to part 1: {star1}")

    star2 = ",".join(ing for _, ing in sorted(known_allergens.items()))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 21
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
