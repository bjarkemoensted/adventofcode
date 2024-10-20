from aocd.models import Puzzle
from collections import defaultdict, Counter
from jinja2 import Template
import numpy as np

from aoc.utils import config


def read_template():
    with open(config.solution_template_path, "r") as f:
        template = f.read()

    return template


def make_solution_draft(day: int, year:int, has_examples=False) -> str:
    raw = read_template()
    temp = Template(raw)
    header = make_ascii_header(year=year, day=day)

    render_kwargs = dict(
        input_folder=config.input_folder,
        input_filename=config.input_filename.format(day=day),
        has_examples=has_examples,
        day=day,
        year=year,
        ascii_header=header
    )

    res = temp.render(**render_kwargs)+"\n"

    return res


_snow_symbols = {
    "small": "⸳.ꞏ`",
    "large": "**+•",
    "space": " "
}

_symbol2cat = {}
for cat, symbols in _snow_symbols.items():
    for symbol in set(symbols):
        if symbol in _symbol2cat:
            raise RuntimeError
        _symbol2cat[symbol] = cat
    #


def anneal_snow(m: np.array, rs: np.random.RandomState, iterations_per_site=1.0):
    """Run simulated annealing on a 2D array represnting ASCII snow-art to attain some desirable
    properties like not having similar symbols cluttered together."""

    rows, cols = m.shape

    max_temp = 2.5
    min_temp = 0.2
    temp_steps = 10
    temperatures = np.linspace(min_temp, max_temp, temp_steps, endpoint=True)[::-1]
    n_its = round(iterations_per_site)*rows*cols

    tension_tol = 0.001

    # Define a tension parameter between symbols in various groups. Tension increases energy
    cat_ten = dict(
        small=0.1,
        large=0.7,
        space=-0.1
    )

    def tension(c1, c2):
        """Computes the 'tension' between any 2 symbols"""
        
        # Identical non-sapce symbol have the greatest (1) tension
        if c1 == c2 and c1 != " ":
            return 1.0
        
        # Otherwise, look up
        cat = _symbol2cat[c1]
        if _symbol2cat[c2] == cat:
            return cat_ten[cat]
        else:
            return 0.0
        #
    
    unit_vectors = tuple(np.array(delta) for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    def energy(char, crd):
        """Computes an energy for a given character being at a given site"""
    
        e = 0.0
        decay = 0.8  # How quickly the tension terms fade with distance
        start_space = char == " "

        for vec in unit_vectors:
            # Iterate in all 4 directions from the start site
            x = crd.copy()
            weight_running = 1.0

            # Keep going until new tension terms will have a very small effect
            while weight_running >= tension_tol:
                # Stop if we fall off the array of if we go from a space to non-space character
                x += vec
                falloff = not all(0 <= comp < dim for comp, dim in zip(x, m.shape))
                if falloff:
                    break
                otherchar = m[*x]
                if start_space and otherchar != " ":
                    break

                # Add any tension with the site reached
                mu = tension(char, otherchar)
                e += mu*weight_running
                weight_running *= decay
            #
        
        return e

    
    crds = [np.array([i, j]) for i in range(rows) for j in range(cols)]
    
    for T in temperatures:
        for _ in range(n_its):
            # Grab 2 random coordinates and their characters
            x1, x2 = (crds[i] for i in rs.choice(len(crds), size=2, replace=False))
            c1, c2 = m[*x1], m[*x2]

            # Determine the decrease in energy by swapping the symbols
            e_current = energy(c1, x1) + energy(c2, x2)
            e_perm = energy(c1, x2) + energy(c2, x1)

            # Keep new if doing so decreases energy
            if e_perm < e_current:
                swap = True
            # Otherwise, use the Boltzmann factor as the accept probability of the new state
            else:
                boltzmann_factor = np.exp(-(e_perm - e_current)/T) if T > 0.0 else 0.0
                swap = boltzmann_factor >= rs.uniform(low=0.0, high=1.0)
            
            if swap:
                m[*x1], m[*x2] = m[*x2], m[*x1]
            #
        #
    #


def let_it_snow(seed: int=None, n_lines: int=4, line_length: int=80):
    """Makes some pretty ASCII snow-art"""

    rs = np.random.RandomState(seed=seed)

    # Define the frequencies of small/large snowflake symbols, and spaces
    dist = dict(
        small=0.375,
        large=0.125,
        space=None
    )
    
    # For symbols with frequencies set to None, interpolate the missing fractions equally
    missing = [(k, v) for k, v in dist.items() if v is None]
    if missing:
        missing_proportion = (1.0 - sum(v for v in dist.values() if v is not None)) / len(missing)
        for k, _ in missing:
            dist[k] = missing_proportion

    # Determine probability distribution over each individual symbol
    symbol_priors = defaultdict(lambda: 0.0)
    for category, symbols in _snow_symbols.items():
        delta_p = dist[category]/len(symbols)
        for symbol in symbols:
            symbol_priors[symbol] += delta_p
        #
    
    all_symbols = sorted(symbol_priors.keys())
    probs = [symbol_priors[sym] for sym in all_symbols]
    
    # Put random snow art into an array
    m = rs.choice(a=all_symbols, size=(n_lines, line_length), p=probs, replace=True)
    
    # Use simulated annealing to optimize some function which correlates with prettiness
    anneal_snow(m, rs=rs)

    return m


def make_ascii_header(year: int, day: int, as_comments=True):
    """Makes an ASCII snow-art header for a new puzzle."""

    line_starter = "# " if as_comments else ""
    min_line_len = 80 - len(line_starter)

    p = Puzzle(year, day)
    elems = (
        p.title,
        p.url
    )

    len_ = max(len(s) for s in elems)
    padding = 3
    n_buffer_lines = 1
    line_length = max(min_line_len, len_ + 2 + padding)
    inject = n_buffer_lines*[""] + list(elems) + n_buffer_lines*[""]

    # Generate the snow-art
    seed = int(f"{year}{day:02}")
    snow_lines = let_it_snow(seed=seed, n_lines=len(inject), line_length=line_length)

    # Insert the required text
    res_lines = []
    for i, sl in enumerate(snow_lines):
        s = inject[i]
        line_chars = sl
        if s != "":
            insert_line = " "+inject[i]+" "
            shift = (len(sl) - len(insert_line)) // 2
            for i, char in enumerate(insert_line):
                line_chars[i + shift] = char
            #
        
        line = line_starter+"".join(line_chars)
        res_lines.append(line)
    
    
    assert len({len(line) for line in res_lines}) == 1
    res = "\n".join(res_lines)
    return res


if __name__ == '__main__':
    print("eyyy")
    year, day = 2020, 10
    draft = make_solution_draft(year, day)
    
    p = Puzzle(year, day)
    print(p.title)
    print(p.url)

    s = make_ascii_header(year=2016, day=15)
    
    # snowlines = let_it_snow(n_lines=10, line_length=80, seed=42)
    # snowstring = "\n".join([''.join(line) for line in snowlines])
    # print(snowstring)
    
    print(s)
