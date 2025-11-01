#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:22:33 2020

@author: ahura
"""

import re


def parse(lines):
    res = []
    for line in lines:
        m = re.match(r'(.*) \(contains (.*)\)', line)
        ingr, al = m.groups()
        ingredients = ingr.split(" ")
        allergens = al.split(", ")
        res.append((ingredients, allergens))

    return res

test = '''mxmxvkd kfcds sqjhc nhms (contains dairy, fish)
trh fvjkl sbzzf mxmxvkd (contains dairy)
sqjhc fvjkl (contains soy)
sqjhc mxmxvkd sbzzf (contains fish)'''.split("\n")

with open("input21.txt") as f:
    lines = [line.strip() for line in f]

stuff = parse(lines)
all_allergens = sorted(set(sum([t[1] for t in stuff], [])))
all_ingredients = sorted(set(sum([t[0] for t in stuff], [])))

allergen2il = {al: [] for al in all_allergens}
for ingredients, allergens in stuff:
    for al in allergens:
        allergen2il[al].append(ingredients)

def get_mentioned_in_all(ingredient_lists):
    counts = {}
    for arr in ingredient_lists:
        for elem in arr:
            counts[elem] = counts.get(elem, 0) + 1
        #
    in_all = [k for k, v in counts.items() if v == len(ingredient_lists)]
    return in_all

### Star 1
remaining_allergens = set(all_allergens)
allergen2determined_ingredient = {}
determined_ingr = set([])
while remaining_allergens:
    newly_resolved = {}
    for allergen in list(remaining_allergens):
        mentioned_ingredients = get_mentioned_in_all(allergen2il[allergen])
        potential_matching_ingredients = [ingr for ingr in mentioned_ingredients
                             if ingr not in determined_ingr]
        found_match = len(potential_matching_ingredients) == 1
        if found_match:
            matching_ingredient = potential_matching_ingredients[0]
            determined_ingr.add(matching_ingredient)
            allergen2determined_ingredient[allergen] = matching_ingredient
            remaining_allergens.remove(allergen)
        #
    #

n_mentions = 0
for ingredients, _ in stuff:
    n_mentions += sum(elem not in determined_ingr for elem in ingredients)

print(n_mentions)

### Star 2
ordered = [tup[1] for tup in sorted(allergen2determined_ingredient.items(),
                                    key=lambda t: t[0])]
canonical_list = ",".join(ordered)
print(canonical_list)