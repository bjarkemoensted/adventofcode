# ·``*·` .·  · *`   · `   .·+`· *   ·   ` · .·   ` · * ·`      · . ·`•· *·`   ·`
# `·*..  ·`     .  ·`·*.    `   Space Image Format   .·  •`   ·`.   + ·.` · *.· 
# ·.  *+·`  .· `  ·    https://adventofcode.com/2019/day/8  ·+.  · •.·     `·* ·
# .  ·+ `  ·`. · •. +` ·  · `* ·  `·*+     .·` +·   · `.  · `  *·  .·. `·`*+ ·`.

from enum import IntEnum

import numpy as np
from aococr import aococr
from numpy.typing import NDArray


class Colors(IntEnum):
    BLACK = 0
    WHITE = 1
    TRANSPARENT = 2


def parse(s: str) -> list[int]:
    return list(map(int, s))


def partition_into_layers(img: list[int], width: int, height: int) -> NDArray[np.int8]:
    """Takes the input image data, and returns an (h, w, n) array, where n is
    the number of layers"""
    n_pixels = width*height
    parts = []
    for i in range(0, len(img), n_pixels):
        layer = np.array(img[i:i+n_pixels], dtype=np.int8).reshape((height, width))
        parts.append(layer)
    
    res = np.stack(parts, axis=0)
    return res


def verification_code(layers: NDArray[np.int8]) -> int:
    """Determines the verification code (product of white and transparent pixels
    in the layer with the fewest black pixels)"""
    ind = (layers == Colors.BLACK).sum(axis=(1, 2)).argmin()
    n_white = (layers[ind] == Colors.WHITE).sum()
    n_transparent = (layers[ind] == Colors.TRANSPARENT).sum()
    return n_white*n_transparent


def decode_image(layers: NDArray[np.int8]) -> str:
    """Decode the BIOS password shown in the image"""
    # Find layer index for the first non-transparent pixel at each position
    mask = layers != Colors.TRANSPARENT
    idx = mask.argmax(axis=0)

    # Condense into an array
    _, n, m = layers.shape
    i = np.arange(n)[:, None]
    j = np.arange(m)
    decoded = layers[idx, i, j]
    
    # Parse into a string
    res = aococr(decoded, pixel_on_off_values=(Colors.WHITE, Colors.BLACK))
    return res


def solve(data: str) -> tuple[int|str, ...]:
    img = parse(data)
    layers = partition_into_layers(img, width=25, height=6)

    star1 = verification_code(layers)
    print(f"Solution to part 1: {star1}")

    star2 = decode_image(layers)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 8
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
