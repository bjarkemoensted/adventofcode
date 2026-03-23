# ·* *·  .  ·`   ··* ·`    *  ·.` ·   `.*     ·  *  * ·· `   · `.  + ·` *. ` `··
# `.+·  ·`   ·.   `·  · * `  · +  Beacon Scanner   ·     ·     ·  ·`·* .`·    *.
# ·` ·.· * ` +·  *  ·  https://adventofcode.com/2021/day/19  `·  ·* ` ·  +  ·.`·
# .··   `· +  * · . `·*   · ` .·+* · ·  `  .`· ·. *·`    *.· · +`   .•·`· .·`*·`


import itertools
from dataclasses import dataclass, field
from functools import cache
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

dtype = np.int_
BEACONS_THRESHOLD = 12


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
    
    res = np.stack(parts, dtype=dtype)
    return res


def distinct_rotations(arr: NDArray[np.int_]) -> NDArray[np.int_]:
    """Generate all 90 degree rotations of a matrix"""

    rotations = _all_rotation_matrices()
    prod = (rotations @ arr.T)
    res = prod.transpose(0, 2, 1)
    
    return res


def pairwise_differences(a: NDArray[np.int_], b: NDArray[np.int_]):
    """Given two arrays of shapes (Na x d) and (Nb x d), computes all pairwise differences
    into an array of shape (Na*Nb x d)."""

    _, dim = a.shape
    assert len(a.shape) == len(b.shape) and b.shape[-1] == dim

    diffs = a[None, :, :] - b[:, None, :]
    res = diffs.reshape(-1, dim)
    return res


@dataclass
class Scan:
    """Represents a scan of the ocean floor. Holds the scanner's ID, and
    the recorded distances to beacons, along with all 3D rotations of the beacons."""

    beacons: NDArray[np.int_]
    id_: int
    rotations: NDArray[np.int_] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rotations = distinct_rotations(self.beacons)
    
    def transform(self, rotation: int, translation: NDArray[np.int_]) -> NDArray[np.int_]:
        """Returns the beacons as they appear after a rotation and translation."""
        res = self.rotations[rotation].copy() + translation
        return res
    #

    @property
    def dim(self) -> int:
        _, d = self.beacons.shape
        return d


def parse(s: str) -> list[Scan]:
    parts = s.split("\n\n")
    res = []
    for part in parts:
        lines = part.splitlines()
        id_ = int(lines[0].split("scanner ")[-1].split()[0])
        arr = np.array([[int(s) for s in line.split(",")] for line in lines[1:]], dtype=dtype)
        scanner = Scan(beacons=arr, id_=id_)
        res.append(scanner)
    
    return res


def find_offset(a: NDArray[np.int_], b: NDArray[np.int_]) -> NDArray[np.int_]|None:
        """Attempts to determine a 3D vector by which to displace b, such that it lines up with a.
        If no such vector exists, such that the threshold for the number of aligned points is met,
        None is returned."""

        _, dim = a.shape
        assert len(a.shape) == len(b.shape) and b.shape[-1] == dim

        # Find all pairwise distances
        pairwise = pairwise_differences(a, b)

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
        if not all(n_match >= BEACONS_THRESHOLD for n_match in n):
            return None
        
        res = np.array(displacement, dtype=dtype)

        return res
    #


def determine_alignment(a: NDArray[np.int_], b: NDArray[np.int_]) -> Iterator[tuple[int, NDArray[np.int_]]]:
    """Given scan arrays a and b, determine the rotation and translation of b, such that a sufficient
    number of beacons overlap with a.
    Array a should be shape (N, 3) and b (24, N, 3) - 24 3D rotations, of N 3D coordinate vectors.
    The result is a tuple of the rotation index, and the translation vector.
    This means if the result is i, x, and if B is the scanner object containing array b, we can
    align it with a by taking B.rotations[i] + x.
    If no alignment is possible, None is returned."""

    assert len(a.shape) == 2 and len(b.shape) == 3
    
    # Go over all rotations of b
    for i, arr in enumerate(b):
        x = find_offset(a, arr)
        if x is not None:
            assert isinstance(x, np.ndarray)
            yield i, x
        #
    #


def align_scanners(*scans: Scan) -> dict[int, tuple[int, NDArray[np.int_]]]:
    """Determine how the scanner must be transformed in order to form a coherent image.
    This works by starting with scanner 0 as a reference, then doing a BFS-style approach, determining
    if and how each of the remaining scanner can be rotated and shifted to achieve a sufficient
    overlap in beacons.
    Returns: A dictionary mapping each scanner ID to a tuple (i, arr), with i indicating the scanner's rotation,
    and arr the translation. This can be passed to the Scan.tranform method to obtain the corrected
    beacon positions."""

    # Store scanners under their ID
    d_scans = {s.id_: s for s in scans}
    
    # Start with the 'reference scanner' (ID 0), with no rotation or translation
    reference = d_scans[0]
    aligned = {reference.id_: (0, np.array([0 for _ in range(reference.dim)], dtype=dtype))}
    
    # Keep looking for remaining scanners that overlap with newly added ones
    front = {0}

    while front:
        next_ = set()
        for id_ in sorted(front):
            # Transform the newly added scanner so its coordinates are relative to the reference scanner
            i, x = aligned[id_]
            known_beacons = d_scans[id_].transform(i, x)

            # Look at each remaining (still unaligned) scanner
            for other, b in d_scans.items():
                if other in aligned:
                    continue
                
                # Look for a way to match with the newly added scanner
                trans = determine_alignment(known_beacons, b.rotations)
                try:
                    # If successful, store the alignment and use the new scanner for next iteration
                    aligned[other] = next(trans)
                    next_.add(other)
                except StopIteration:
                    continue  # If no match is possible, proceed to the nest remaining scanner
                #
            #
        front = next_
    
    assert len(aligned) == len(scans)
    return aligned


def solve(data: str) -> tuple[int|str, ...]:
    scans = parse(data)
    alignment = align_scanners(*scans)
    beacons = np.vstack([scan.transform(*alignment[scan.id_]) for scan in scans])

    star1 = len(np.unique(beacons, axis=0))
    print(f"Solution to part 1: {star1}")

    scanner_positions = np.vstack([pos for _, pos in alignment.values()])
    scanner_dists = pairwise_differences(scanner_positions, scanner_positions)
    star2 = np.abs(scanner_dists).sum(axis=-1).max()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 19
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
