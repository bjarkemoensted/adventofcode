# ·* *·  .  ·`   ··* ·`    *  ·.` ·   `.*     ·  *  * ·· `   · `.  + ·` *. ` `··
# `.+·  ·`   ·.   `·  · * `  · +  Beacon Scanner   ·     ·     ·  ·`·* .`·    *.
# ·` ·.· * ` +·  *  ·  https://adventofcode.com/2021/day/19  `·  ·* ` ·  +  ·.`·
# .··   `· +  * · . `·*   · ` .·+* · ·  `  .`· ·. *·`    *.· · +`   .•·`· .·`*·`


import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


def make_rotation_matrix(alpha: float, beta: float, gamma: float) -> NDArray[np.floating]:
    """Make a 3D rotation matrix"""
    cos, sin = np.cos, np.sin

    M = np.array(
        [
            [
                cos(alpha)*cos(beta), cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma),
                cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma)
            ],
            [
                sin(alpha)*cos(beta), sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma),
                sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma)
            ],
            [
                -sin(beta), cos(beta)*sin(gamma),
                cos(beta)*cos(gamma)
            ]
        ],
        dtype=float
    )
    
    return M


def _coords_from_array(arr: NDArray[np.int_]) -> list[tuple[int, int, int]]:
    """Get the coordinates from each row in an array, as a list of tuples."""
    coords = (tuple(map(int, row) for row in arr))
    res = [(a, b, c) for a, b, c in coords]
    return res


@cache
def _all_rotation_matrices() -> NDArray[np.int_]:
    """Generate all distinct 3D rotation matrices for n*pi/2 rotations"""

    parts = []
    seen: set[bytes] = set()

    for angles in itertools.product((n*np.pi/2 for n in range(4)), repeat=3):
        M = make_rotation_matrix(*angles)
        M_int = M.astype(int)
        # Make sure we round correctly
        assert np.allclose(M, M_int)

        # Ignore matrix if it's already been generated
        key = M_int.tobytes()
        if key in seen:
            continue
        seen.add(key)

        parts.append(M_int)
    
    res = np.stack(parts)
    return res



def distinct_rotations(arr: NDArray[np.int_]) -> NDArray[np.int_]:
    """Generate all 90 degree rotations of a matrix"""

    parts = []

    for M in _all_rotation_matrices():
        x = np.dot(arr, M.T)
        parts.append(x)
    
    res = np.stack(parts)
    return res


@dataclass
class Scanner:
    beacons: NDArray[np.int_]
    id_: int
    rotations: NDArray[np.int_] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rotations = distinct_rotations(self.beacons)
    #


def parse(s: str) -> list[Scanner]:
    parts = s.split("\n\n")
    res = []
    for part in parts:
        lines = part.splitlines()
        id_ = int(lines[0].split("scanner ")[-1].split()[0])
        arr = np.array([[int(s) for s in line.split(",")] for line in lines[1:]])
        scanner = Scanner(beacons=arr, id_=id_)
        res.append(scanner)
    
    return res


class MapAssembler:
    """Helper class for efficiently (ish) determining whether and how scan arrays can line up
    such that the required number of beacon points overlap."""

    def __init__(self) -> None:
        self._cache: dict[tuple[bytes, bytes], NDArray[np.int_]|None] = dict()
        self.threshold = 12
        self.dim = 3
    
    @staticmethod
    def _key(a: NDArray[np.int_], b: NDArray[np.int_]) -> tuple[bytes, bytes]:
        ka, kb = (arr.tobytes() for arr in (a, b))
        return ka, kb

    def _find_offset(self, a: NDArray[np.int_], b: NDArray[np.int_]) -> NDArray[np.int_]|None:
        """Attempts to determine a 3D vector by which to displace b, such that it lines up with a.
        If no such vector exists, such that the threshold for the number of aligned points is met,
        None is returned."""

        # Find all pairwise distances
        diffs = a[None, :, :] - b[:, None, :]
        pairwise = diffs.reshape(-1, self.dim)

        # Find the displacement along each direction which occurs most frequently
        inds = []
        n = []
        displacement = []
        for col in pairwise.T:
            vals, counts = np.unique(col, return_counts=True)
            ind = np.argmax(counts)
            inds.append(ind)
            n.append(int(counts[ind]))
            displacement.append(vals[ind])
        
        # The arrays can line up iff the most common displacement meets threshold in all directions
        if not all(n_match >= self.threshold for n_match in n):
            return None
        
        res = np.array(displacement)

        # Double check that we meet the threshold after adding the result to b
        overlaps_after_shift = set(_coords_from_array(a)).intersection(_coords_from_array(b + res))
        assert len(overlaps_after_shift) >= self.threshold

        return res

    def find_offset(self, a: NDArray[np.int_], b: NDArray[np.int_]) -> NDArray[np.int_]|None:
        key = self._key(a, b)
        if key in self._cache:
            return self._cache[key]
    
        res = self._find_offset(a, b)
        self._cache[key] = res
        return res

    def determine_alignment(self, a: NDArray[np.int_], b: NDArray[np.int_]) -> Iterator[tuple[int, NDArray[np.int_]]]:
        """Given scan arrays a and b, determine the rotation and translation of b, such that a sufficient
        number of beacons overlap with a.
        Array a should be shape (N, 3) and b (24, N, 3) - 24 3D rotations, of N 3D coordinate vectors.
        The result is a tuple of the rotation index, and the translation vector.
        This means if the result is i, x, and if B is the scanner object containing array b, we can
        align it with a by taking B.rotations[i] + x.
        If no alignment is possible, None is returned."""

        assert len(a.shape) == 2 and len(b.shape) == 3 and a.shape[-1] == b.shape[-1] == self.dim
        
        # Go over all rotations of b
        for i, arr in enumerate(b):
            x = self.find_offset(a, arr)
            if x is not None:
                yield i, x
            #
        
    def scanners_align(self, a: Scanner, b: Scanner) -> bool:
        """Determines whether it's possible to align scanners a and b"""
        try:
            next(self.determine_alignment(a.beacons, b.rotations))
            return True
        except StopIteration:
            return False
        #


assembler = MapAssembler()


def _connect_scans(scanners: dict[int, Scanner]) -> dict[int, list[int]]:
    """Given the scans, determine which scans connect to which.
    In the resulting dict, the presence of d[a] = b, means that
    scanner a contains a number of beacons (at least the threshold) which
    overlap with some rotation of scan b."""
    
    connections: dict[int, set[int]] = defaultdict(set)
    
    ids = sorted(scanners.keys())
    for ka in ids:
        for kb in ids:
            a, b = scanners[ka], scanners[kb]
            if a.id_ == b.id_:
                continue
            if assembler.scanners_align(a, b):
                connections[a.id_].add(b.id_)
            #
        #
    
    return {k: sorted(v) for k, v in connections.items()}
    

def assemble_image(*scanners: Scanner) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    d = {scanner.id_: scanner for scanner in scanners}

    ids = sorted(d.keys())
    assert len(ids) == len(scanners)

    links = _connect_scans(d)

    ref = ids[0]
    resolved = {ref: d[ref].beacons}
    front = [ref]

    offsets = []

    while front:
        next_: list[int] = []
        for id_ in front:
            if id_ not in links:
                continue
            beacons = resolved[id_]
            next_ids = links[id_]

            for next_id in next_ids:
                next_scanner = d[next_id]
                if next_id in resolved:
                    continue
                next_.append(next_id)

                alignment = next(assembler.determine_alignment(beacons, next_scanner.rotations))

                i, x = alignment
                offsets.append(x)
                next_beacons = next_scanner.rotations[i] + x
                resolved[next_id] = next_beacons

                meh_a = set(_coords_from_array(beacons))
                meh_b = set(_coords_from_array(next_beacons))
                overlap = list(meh_a.intersection(meh_b))

                assert len(overlap) >= 12  # !!!

            
        front = next_
    
    all_beacons = np.vstack(list(resolved.values()))
    translations = np.vstack(offsets)

    return translations, all_beacons



def solve(data: str) -> tuple[int|str, ...]:
    scanners = parse(data)
    translations, beacons = assemble_image(*scanners)

    star1 = len(np.unique(beacons, axis=0))
    print(f"Solution to part 1: {star1}")

    pairwise_dists = translations[:, None, :] - translations[None, :, :]    
    star2 = np.abs(pairwise_dists).sum(axis=-1).max()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 19
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
