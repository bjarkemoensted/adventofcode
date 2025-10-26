from dataclasses import dataclass
from functools import partial
import numpy as np
from numpy.typing import NDArray
import typing as t


# Some special characters mess up the line widths as displayed in VSCode
_forbidden_ords = (42895, 11827)


# Symbols to use for ascii snow art, and their weights (proportional to expected frequency)
_default_symbols: dict[str, float] = {
    " ": 4.5,
    "·": 2,
    ".": 1,
    "`": 1,
    "*": 0.7,
    "+": 0.3,
    "•": 0.15
}

# These symbols are considered 'large'
_default_large_symbols = ("*", "+", "•")


@dataclass
class Tension:
    """Helper class for computing the 'tension' between two characters, depending on their values and locations in
    some array of characters."""

    # tension energies for various kinds of characters
    tension_small: float=0.2
    tension_large: float=0.7
    tension_space: float=-0.1
    tension_ident: float=2.0
    tension_tolerance: float = 0.01
    large_symbols: t.Iterable[str]=()
    unit_vectors: t.ClassVar = tuple(np.array(delta) for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    
    def __post_init__(self):
        # Convert large symbols into a set, for better lookup
        self._large = set(self.large_symbols)
    
    def _tension(self, c1: str, c2: str) -> float:
        """Computes the tension between two characters"""
        
        # If characters match exactly, look up the corresponding energy
        if c1 == c2:
            if c1 == " ":
                return self.tension_space
            else:
                return self.tension_ident
            #
        
        # Otherwise, base tension on matching categories (small-small, large-large)
        is_large = tuple(c in self._large for c in (c1, c2))
        if all(is_large):
            return self.tension_large
        elif not any(is_large):
            return self.tension_small
        
        return 0.0
    
    def __call__(self, m: NDArray[np.str_], char: str, i: int, j: int) -> float:
        """Computes the energy for a character at the specified site"""
        
        e = 0.0
        decay = 0.8  # How quickly the tension terms fade with distance
        start_space = char == " "

        for vec in self.unit_vectors:
            # Iterate in all 4 directions from the start site
            x = np.array([i, j])
            weight_running = 1.0

            # Keep going until new tension terms will have a very small effect
            while weight_running >= self.tension_tolerance:
                # Stop if we fall off the array of if we go from a space to non-space character
                x += vec
                falloff = not all(0 <= comp < dim for comp, dim in zip(x, m.shape))
                if falloff:
                    break
                otherchar = m[*x]
                if start_space and otherchar != " ":
                    break

                # Add any tension with the site reached
                mu = self._tension(char, otherchar)
                e += mu*weight_running
                weight_running *= decay
            #
        
        return e
    #


class Annealer:
    """Handles simulated annealing of ASCII snow/art."""
    
    def __init__(
            self,
            seed: int|None=None,
            symbols: dict[str, float]|None=None,
            large_symbols: t.Iterable[str]=_default_large_symbols,
            energy_kwargs: dict[str, float]|None=None,
            verbose=False
            ) -> None:
        """ Instantiates an Annealer
        seed: random seed
        symbols: dict of symbols to use, mapping characters to their weights (proportional to frequencies)
        large_symbols: The symbols which are considered large,
        energy_kwargs: optional dictionary of keyword arguments for the Tension class, which computes energies,
        verbose: Whether to output details 
        """
        
        self.verbose = verbose
        kwargs = energy_kwargs or dict()
        self.energy = Tension(large_symbols=large_symbols, **kwargs)
        self.symbols = symbols if symbols is not None else _default_symbols
        if not set(large_symbols).issubset(self.symbols.keys()):
            raise ValueError
        
        bad_syms = [c for c in self.symbols.keys() if ord(c) in _forbidden_ords]
        if bad_syms:
            raise ValueError(f"Attempted to use glyphs that don't monospace well: {', '.join(bad_syms)}")
        
        self.rs = np.random.RandomState(seed=seed)
    
    def vprint(self, *args, **kwargs) -> None:
        if self.verbose:
            print(*args, **kwargs)
    
    def initialize_array(self, rows: int, cols: int) -> NDArray[np.str_]:
        """Initializes an array with the specified dimensions."""
        
        # Use the weights to compute frequencies for each symbol
        weights = self.symbols
        norm_factor = 1.0/sum(weights.values())
        chars = list(weights.keys())
        probs = [weights[char]*norm_factor for char in chars]
        
        # Create the array using random sampling
        shape = (rows, cols)
        m = self.rs.choice(
            a=chars,
            size=shape,
            p=probs,
            replace=True
        )
        self.vprint(f"Created array with shape: {m.shape}")
        
        return m
    
    def _anneal(self, m: NDArray[np.str_], temperatures: t.Iterable[float], n_repeats: int=1) -> None:
        """Performs simulated annealing, modifying the matrix in-place"""
        
        rows, cols = m.shape
        
        crds = [np.array([i, j]) for i in range(rows) for j in range(cols)]
        energy = partial(self.energy, m)

        for T in temperatures:
            reductions = [0.0 for _ in range(n_repeats)]
            swapped = 0
            
            for i in range(n_repeats):
                # Grab 2 random coordinates and their characters
                x1, x2 = (crds[i] for i in self.rs.choice(len(crds), size=2, replace=False))
                c1, c2 = m[*x1], m[*x2]

                # Determine the decrease in energy by swapping the symbols
                e_current = energy(c1, *x1) + energy(c2, *x2)
                e_perm = energy(c1, *x2) + energy(c2, *x1)

                # Keep new if doing so decreases energy
                energy_reduction = e_current - e_perm
                if energy_reduction > 0.0:
                    swap = True
                # Otherwise, use the Boltzmann factor as the accept probability of the new state
                else:
                    boltzmann_factor = np.exp(-(e_perm - e_current)/T) if T > 0.0 else 0.0
                    swap = boltzmann_factor >= self.rs.uniform(low=0.0, high=1.0)
                    
                if swap:
                    m[*x1], m[*x2] = m[*x2], m[*x1]
                    reductions[i] = energy_reduction
                    swapped += 1
                #
                
            
            self.vprint(f"Temperature {T}, energy reduced by: {sum(reductions):.2f}, {swapped=}")
        
    def __call__(
            self,
            rows: int,
            cols: int,
            max_temp: float = 2.5,
            min_temp: float = 0.2,
            temp_steps: int = 10,
            iterations_per_site=1.0
            ) -> str:
        """Create a snow-art header by running simulated annealing with the specified temperature parameters"""
        
        m = self.initialize_array(rows, cols)
        
        n_repeats = round(iterations_per_site)*rows*cols
        temperatures = np.linspace(min_temp, max_temp, temp_steps, endpoint=True)[::-1]
        
        self._anneal(m=m, temperatures=temperatures, n_repeats=n_repeats)
        
        res = "\n".join(["".join(row) for row in m])
        return res
    #
