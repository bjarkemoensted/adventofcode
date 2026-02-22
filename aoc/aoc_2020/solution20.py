# ·`     ·.  * +·. `. · * ` .    ·* `·.`  ·    . *  `·      ·*` *. ·`· .    ·`·*
# `·*· . ·+ ` ·`     .  · · +`*  Jurassic Jigsaw · +     *.·`.•·· .   ·`+ ·.` ·.
# ·.·   •` *·`   · . ` https://adventofcode.com/2020/day/20  ·  . ·       ` .·` 
# . ·*`· . `+·    ·   `· `•  · ·+ `·. *  ·` · *   `·  .·*`  .  `+·*.·       +`.·

import functools
import typing as t
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

_pixel_on = "#"
_pixel_off = "."


# The sea monster we're looking for in part 2
_sea_monster = (
"""                  # 
#    ##    ##    ###
 #  #  #  #  #  #   
"""
)


@functools.cache
def compute_key_from_edge(*bits: int) -> int:
    """Given the edge on a n array with ones and zeros, computes an integer which is unique
    for that edge, by interpreting values at the edge as a binary number."""
    res = sum(b*2**i for i, b in enumerate(reversed(bits)))
    return int(res)


def signature(arr: NDArray[np.int_]) -> tuple[int, int, int, int]:
    """Tuple of the keys representing all four edges (NESW) of a tile"""
    
    edges = [arr[0, :], arr[:, -1], arr[-1, :], arr[:, 0]]
    n, e, s, w = (compute_key_from_edge(*edge) for edge in edges)
    return n, e, s, w



def generate_transformations(m: NDArray[np.int_]) -> t.Iterator[NDArray[np.int_]]:
    """Takes a 2D array representing a tile.
    Generates each transformation obtained by flipping/rotating the present.
    Takes an nxn array and returns an nxnxm array, where m is the number of transformations"""
    
    for n_rot in range(4):
        m_rotated = np.rot90(m, n_rot)
        for flipped in (m_rotated, np.fliplr(m_rotated)):
            yield flipped
        #


class Tile:
    """Represents a tile in the jigsaw puzzle.
    To allow efficient computations, the most important aspects of the tile are encapsulated
    in the .signature property, which is an 8x4 numpy array.
    Each of the 8 rows represents a transformation (combination of rotating and flipping),
    and each column represents the 4 'edge keys' of that transformation.
    An edge key is an integer which captures the shape of an edge, by interpreting the pixel
    values (on/off) as the binary representation of an integer. If two edges of two tiles
    have the same key, that means those two edges can line up."""

    def __init__(self, id_: int, pixels: NDArray[np.int_]) -> None:
        self.id = id_
        self.arr = pixels.copy()
        self.transformations = np.array(list(generate_transformations(self.arr)))
        n_transformations = len(self.transformations)
        n_sides = 4

        # Define an array with the key at each edge (NESW) for each transformation
        self.signature = np.full((n_transformations, n_sides), -1, dtype=int)
        seen = set()
        for i, trans in enumerate(generate_transformations(self.arr)):
            # Only keep distinct transformations
            key = tuple(tuple(row) for row in trans)
            if key in seen:
                continue
            seen.add(key)
            for j, edge in enumerate(signature(trans)):
                self.signature[i, j] = edge
            #
        
        self.keys = {int(k) for k in self.signature.flat}
    
    def __repr__(self):
        key = ", ".join(map(str, self.signature[0]))
        return f"{self.__class__.__name__}(id={self.id}, key=({key}))"

    @property
    def ascii(self) -> str:
        """Display tile as ascii, using # and . symbols. For debugging and stuff"""
        map_ = {0: _pixel_off, 1: _pixel_on}
        res = "\n".join("".join(map_[char] for char in row) for row in self.arr)
        return res


def parse(s: str) -> list[Tile]:
    res: list[Tile] = []
    for part in s.split("\n\n"):
        lines = part.splitlines()
        id_ = int(lines[0].split("Tile ")[1].split(":")[0])
        stringarr = np.array([list(line) for line in lines[1:]])
        assert all(val in (_pixel_on, _pixel_off) for val in stringarr.flat)

        pixels = (stringarr == _pixel_on).astype(int)
        tile = Tile(id_=id_, pixels=pixels)
        res.append(tile)

    return res


def complete(
        signatures: NDArray[np.int_],
        map_: NDArray[np.int_],
    ):
    """Attempt to complete the puzzle. Inputs are the signatures of all tiles (nx8x4), and a map_ (dx2), representing
    the tile and transformation at each site.
    Works by scanning across the map for sites that have no tile. Looks to the north and east for any neighbors,
    and determines which (if any) transformations of the remaining tiles line up with the neighbors.
    The valid choice is inserted in-place.
    Assumes that no such position will have multiple valid choices for tiles - this seems to be an input constraint,
    but could easily be addressed with brute force, by continuing for each valid choice separately.
    In cases where no valid choice exists, the function terminates normally, so check that no all tiles have
    been placed before accepting the result."""
    
    for i, j in list(np.ndindex(map_.shape[:2])):
        if map_[i, j, 0] != -1:
            continue  # Skip sites that already contain a tile
        
        # Require that the next tile is yet unused
        used = np.unique(map_[:, :, 0])
        valid_i = ~np.isin(np.arange(signatures.shape[0]), used)
        mask = valid_i[:, None]

        # If there's a tile to the west, also require its eastern face match this tile's western face
        if j > 0:
            key_e = signatures[*map_[i, j-1], 1]  # neighbor's eastern face (index 1)
            c_left = (signatures[:, :, 3] == key_e)[:,:]  # match with  western face (index 3)
            mask = mask & c_left

        # If there's a tile to the north, also require its southern face match this tile's northern face
        if i > 0:
            key_s = signatures[*map_[i-1, j], 2]  # neighbor's southern face (index 2)
            c_up = (signatures[:, :, 0] == key_s)[:,:]  # match with  northern face (index 0)
            mask = mask & c_up

        # match with neighbor
        candidates = np.argwhere(mask)
        
        if len(candidates) > 1:
            raise RuntimeError("Got multiple possible neighbors")
        
        elif len(candidates) == 0:
            return

        # Insert the valid tile
        map_[i, j] = candidates
    #


class Jigsaw:
    """This is responsible for holding the tiles of the image puzzle.
    For efficient computation, this class stores the tile 'signature' - a large numpy array,
    of shape nx8x4. n is the number of tiles, 8 is the number of transformations, and 4 is, for each
    tile*transformation, the 'keys' representing the 4 edges of the tile."""

    def __init__(self, tiles: list[Tile]) -> None:
        self.tiles = tiles
        self.signatures = np.array([tile.signature for tile in self.tiles])
        self.n_tiles = len(self.tiles)
        self.side_len = round(self.n_tiles**.5)
        assert self.side_len**2 == self.n_tiles

        self.shape = (self.side_len, self.side_len)
        _tile_shapes = {tile.arr.shape for tile in self.tiles}
        assert len(_tile_shapes) == 1
        th, tw = _tile_shapes.pop()
        assert th == tw
        self.tile_width = tw

        self.ids = np.array([tile.id for tile in self.tiles], dtype=int)
        self.indices = np.array(range(self.n_tiles), dtype=int)

        # Map each key to the tile indices with that edge
        self.key_to_inds: dict[int, set[int]] = defaultdict(set)
        for i, tile in enumerate(self.tiles):
            for k in tile.keys:
                self.key_to_inds[k].add(i)
            #

        # Map each tile index to the indices it can line up with
        self.matches = [
            set.union(*(self.key_to_inds[k] for k in tile.keys)) - {i}
            for i, tile in enumerate(self.tiles)
        ]

    def determine_arrangement(self) -> NDArray[np.int_]:
        """Finds the arrangement of tiles where all edges fit together.
        The result is a dxdx2 array, where each element i, j, k
        represents the tile (k=0) and transformation (k=1)."""

        # Holds the layout of the puzzle. For each tile location, there's room for the tile index and transformation
        empty = np.full((self.side_len, self.side_len, 2), -1, dtype=int)

        for ind_start in sorted(self.indices, key=lambda i: len(self.matches[i])):
            for transind in range(len(self.tiles[ind_start].transformations)):
                arrangement = empty.copy()
                arrangement[0, 0] = np.array([ind_start, transind])
                complete(
                    signatures=self.signatures,
                    map_=arrangement
                )
                finished = np.all(arrangement[:, :, 0] != -1)
                if finished:
                    return arrangement
                #
            #
        raise RuntimeError

    def resolve_to_id_grid(self, arrangement: NDArray[np.int_]) -> NDArray[np.int_]:
        """Takes a way to arrange the puzzle pieces (nxnx2) array representing the tile index,
        and transformation of each tile.
        Returns a grid of the tile IDs at each site;"""

        res = self.ids[arrangement[:,:, 0]]
        return res

    def assemble_picture(self, arrangement: NDArray[np.int_]) -> NDArray[np.int_]:
        """Takes an arrangement of pieces (tile index and transformation).
        Returns the pieces assembled into a single image"""

        n = self.tile_width - 2  # width of each piece (the edges are dropped)
        d = self.side_len  # image width in terms of number of pieces
        parts = np.full((d, n, d, n), -1, dtype=int)

        for i, j in np.ndindex(self.shape):
            # Look up the tile and transformation
            tile, transformation  = arrangement[i, j]
            part = self.tiles[tile].transformations[transformation]
            parts[i, :, j, :] = part[1:-1, 1:-1]
        
        assert not np.any(parts == -1)

        res = parts.reshape(d*n, d*n)
        return res


def count_sea_monsters(image: NDArray[np.int_]) -> int:
    """Counts the number of sea monsters in the image.
    Uses all transformation of the sea monster drawing, and returns the first
    non-zero monster count."""
    
    # Flip/rotate the monster instead of the whole map
    sea_monster_arr = np.array([list(line) for line in _sea_monster.splitlines()])
    space = np.array(sea_monster_arr != _pixel_on)

    for transformed in generate_transformations(space):
        n = 0

        height, width = transformed.shape
        search_space = tuple(
            dim - frame for dim, frame in zip(image.shape, transformed.shape, strict=True)
        )
        for i, j in np.ndindex(search_space):
            part = image[i:i+height, j:j+width]
            # The section should have values 1 or line up with space in the sea monster shape
            match = np.all(part | transformed)
            if match:
                n += 1
            #
        if n:
            return n
        #
    
    return 0


def solve(data: str) -> tuple[int|str, ...]:
    tiles = parse(data)

    jigsaw = Jigsaw(tiles)

    arrangement = jigsaw.determine_arrangement()
    id_grid = jigsaw.resolve_to_id_grid(arrangement)
    star1 = id_grid[0, 0]*id_grid[0, -1]*id_grid[-1, -1]*id_grid[-1, 0]
    print(f"Solution to part 1: {star1}")

    image = jigsaw.assemble_picture(arrangement)
    n_monsters = count_sea_monsters(image)
    roughness = image.sum() - n_monsters*sum(char == _pixel_on for char in _sea_monster)

    star2 = roughness
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 20
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
