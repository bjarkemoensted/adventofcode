from collections import Counter, defaultdict
import numpy as np


# Letter maps from: https://github.com/mstksg/advent-of-code-ocr/blob/main/src/Advent/OCR/LetterMap.hs


#######################################################################################################################
##                                        Small (6x4 pixel) font                                                     ##
#######################################################################################################################

smol_chars = "ABCEFGHIJKLOPRSUYZ"
smol_widths = defaultdict(lambda: 4, {"I": 3, "Y": 5})

smol = """
.##..###...##..####.####..##..#..#.###...##.#..#.#.....##..###..###...###.#..#.#...#.####
#..#.#..#.#..#.#....#....#..#.#..#..#.....#.#.#..#....#..#.#..#.#..#.#....#..#.#...#....#
#..#.###..#....###..###..#....####..#.....#.##...#....#..#.#..#.#..#.#....#..#..#.#....#.
####.#..#.#....#....#....#.##.#..#..#.....#.#.#..#....#..#.###..###...##..#..#...#....#..
#..#.#..#.#..#.#....#....#..#.#..#..#..#..#.#.#..#....#..#.#....#.#.....#.#..#...#...#...
#..#.###...##..####.#.....###.#..#.###..##..#..#.####..##..#....#..#.###...##....#...####
"""

#######################################################################################################################
##                                       Larger (10x6 pixel) font                                                    ##
#######################################################################################################################

lorge_chars = "ABCEFGHJKLNPRXZ"
lorge_widths = defaultdict(lambda: 6)

lorge = """
..##...#####...####..######.######..####..#....#....###.#....#.#......#....#.#####..#####..#....#.######
.#..#..#....#.#....#.#......#......#....#.#....#.....#..#...#..#......##...#.#....#.#....#.#....#......#
#....#.#....#.#......#......#......#......#....#.....#..#..#...#......##...#.#....#.#....#..#..#.......#
#....#.#....#.#......#......#......#......#....#.....#..#.#....#......#.#..#.#....#.#....#..#..#......#.
#....#.#####..#......#####..#####..#......######.....#..##.....#......#.#..#.#####..#####....##......#..
######.#....#.#......#......#......#..###.#....#.....#..##.....#......#..#.#.#......#..#.....##.....#...
#....#.#....#.#......#......#......#....#.#....#.....#..#.#....#......#..#.#.#......#...#...#..#...#....
#....#.#....#.#......#......#......#....#.#....#.#...#..#..#...#......#...##.#......#...#...#..#..#.....
#....#.#....#.#....#.#......#......#...##.#....#.#...#..#...#..#......#...##.#......#....#.#....#.#.....
#....#.#####...####..######.#.......###.#.#....#..###...#....#.######.#....#.#......#....#.#....#.######
"""


def _to_array(data: str|list|np.ndarray):
    """Takes some characters in either a string format, a list of lists of characters, or a numpy array.
    Returns the data as a numpy array."""

    if isinstance(data, np.ndarray):
        m = data
    elif isinstance(data, str):
        cleaned = data.strip()
        list_of_lists = [list(line) for line in cleaned.splitlines()]
        m = _to_array(data=list_of_lists)
    elif isinstance(data, list):
        if len({len(row) for row in data}) != 1:
            raise ValueError(f"Converting list of lists to numpy array, but rows have inconsistent lengths")
        m = np.array(data)
    else:
        raise TypeError(f"Unsupported type: {type(data)}.")
    
    return m


def _arr_to_str(m: np.array, char_replacements: dict=None) -> str:
    """Converts array to a string, with rows separated by newlines.
    Takes an optional dict for replacement characters."""

    lines = [''.join([char_replacements.get(char, char) for char in line]) for line in m]
    res = "\n".join(lines)
    return res


def display(m: np.array):
    """Prints an array of characters in a way that looks good on the terminal.
    Replaces '.' with empty space " " to make reading easier."""

    replace = {".": " "}
    s = _arr_to_str(m=m, char_replacements=replace)
    print(s)


class Scanner:
    """Helper class for scanning across a 2D array to make pattern matching easier.
    Works by sliding a window spanning all rows across the array.
    Maintains the index of the left edge of the window, and exposes functionality to
    move the index forward, and grab the subarray corresponding to windows of a given pixel width
    starting at the index."""

    def __init__(self, data: np.ndarray, spacing: int=0):
        """data: numpy string array with values "#" (on) and "." (off).
        spacing: Assumed spacing between characters. Defaults to 0."""

        self.m = data
        _, self.edge = self.m.shape
        self.ind = 0
        self.spacing = spacing
    
    def skip_ahead(self, n_pixels=1):
        """Skips n pixels ahead towards the right"""
        self.ind += n_pixels

    def peek(self, window_width: int):
        """Returns an array corresponding to a windows n pixels wide, starting from the index.
        Does not alter the current index."""
        
        left = self.ind
        right = self.ind + window_width
        
        # Numpy just stops at the edge which can be misleading, so throw an error if the right edge falls off
        if right > self.edge:
            raise IndexError(f"Window right edge at {right} falls off (array width: {self.edge})")
        
        snippet = self.m[:, left:right]
        return snippet
    
    def pop(self, window_width: int):
        """Returns contents of a window n pixels wide starting from index.
        Moves index a corresponding distance forward."""

        res = self.peek(window_width=window_width)
        self.skip_ahead(window_width + self.spacing)
        return res
    
    def match(self, target: np.ndarray, skip_ahead_on_match=True) -> bool:
        """Accepts a numpy array and returns a bool indicating whether a window starting at
        the scanner's current index matches the input.
        If skip_ahead_on_match (default: True), the index is moved forward by the window width."""

        _, window_width = target.shape
        try:
            snippet = self.peek(window_width)
            # Explicitly check that the dimensions match as well
            is_match = (snippet.shape == target.shape) and np.all(snippet == target)
        except IndexError:
            is_match = False
        
        if is_match and skip_ahead_on_match:
            self.pop(window_width)

        return is_match
    
    def done(self) -> bool:
        """Bool indicating whether the scanner has any data left."""
        res = self.ind >= self.edge
        return res


def _parse_glyphs(characters, widths: dict, ascii_glyphs: str):
    """Parses the ASCII glyphs and iterates over pairs of characters and their corresponding
    ascii art representation."""

    data = _to_array(data=ascii_glyphs)
    scanner = Scanner(data=data, spacing=1)
    for char in characters:
        width = widths[char]
        glyph = scanner.pop(window_width=width)
        yield char, glyph
    #


def load_glyphs(glyph_dimensions_pixels: tuple=None, data: np.ndarray=None):
    """Loads data on ascii glyphs of the specified size.
    If no size is provided, the appropriate size is inferred from the data, by
    looking at the height."""

    smolsize = (6, 4)
    lorgesize = (10, 6)

    if glyph_dimensions_pixels is None:
        if data is None:
            raise ValueError("Either provide glyph size (heigth, width) or data from which to infer is")
        
        # Just look for the glyphs with the same pixel height as the input
        height, _ = data.shape
        if height == 6:
            glyph_dimensions_pixels = smolsize
        elif height == 10:
            glyph_dimensions_pixels = lorgesize
        #
    
    d = {
        smolsize: dict(characters=smol_chars, widths=smol_widths, ascii_glyphs=smol),
        lorgesize: dict(characters=lorge_chars, widths=lorge_widths, ascii_glyphs=lorge)
    }

    try:
        kwargs = d[glyph_dimensions_pixels]
        res = list(_parse_glyphs(**kwargs))
        return res
    except KeyError:
        raise ValueError(f"Couldn't load glyphs for size: {glyph_dimensions_pixels}")




def standardize(m: np.ndarray, pixel_on=None):
    """Standardizes a numpy array representing ascii glyphs.
    Attempts to determine on/off pixel values from the input.
    If pixel_on is provided, the remaining value is assumed to mean off.
    If input only contains the standard ('#' and '.') values, they are assumed
    to hold their usual meanings. Otherwise, it is assumed that the most common character means off,
    and the remaining character means on.
    An error is raised if the input does not contain exactly 2 distinct values."""
    
    target_on = "#"
    target_off = "."


    counts = Counter(m.flat)
    if len(counts) != 2:
        raise ValueError(f"Expected 2 distinct pixel values but got {len(counts)}: {', '.join(counts.keys())}.")

    # Infer pixel on/off values if not specified
    if pixel_on is None:
        # If the standard pixel values are used, assume the usual meaning
        if set(counts.keys()) == {target_on, target_off}:
            return m
        
        # Otherwise, assume the most common value represents off, and the other on.
        pixel_on, pixel_off = sorted(counts.keys(), key=lambda val: counts[val])
    
    # Replace values to get the standard format
    replace = {pixel_on: target_on, pixel_off: target_off}
    rows, cols = m.shape
    res = np.full(m.shape, " ", dtype='<U1')

    for i in range(rows):
        for j in range(cols):
            res[i, j] = replace[m[i, j]]

    return res

def ocr(
        data: str|list|np.ndarray,
        pixel_on: str = None,
        glyph_dimensions_pixels: tuple=None
    ) -> str:
    """Parses the ASCII art representations of letters sometimes encountered in Advent of Code (AoC).
    Whereas most problems have solutions which produce interger outputs, a few output stuff like:

    .##..###...##.
    #..#.#..#.#..#
    #..#.###..#...
    ####.#..#.#...
    #..#.#..#.#..#
    #..#.###...##.

    A human can easily parse the above into "ABC", but it's nice to be able to do programatically.

    This function can parse ascii art-like data like the above into a string.

    data: The ascii art-like data to be parsed. Multiple formats can be used:
        string: Plaintext, with newlines characters separating the lines.
        list of lists, with each element of the inner list being a single character.
        numpy array: 2D string array where each element is a single character. Other values
            (e.g. integer array) will also be attempted interpreted.
    pixel_on: AoC tends to use "#" and "." to represent pixels being on/off, respectively.
        If the input uses different symbols, pixel_on will be interpreted as the pixel being on,
        and converted to "#" when matched against known glyphs. As the data may only contain 2
        distinct pixel values, the remaining value is assumed to mean off.
        If no argument is provided, the values are inferred from the data.
        An error is thrown if the data do not contain exactly 2 distinct pixel values.
    glyph_dimension_pixels (tuple): AoC has featured ascii art stuff with various font sizes.
        A specific font size can be specified here in a (height_in_pixels, width_in_pixels) format.
        The supported values are (6, 4), and (10, 6) - the only ones occurring to my knowledge.
        If not provided, the size is inferred from the data, going by the height, assuming a single
        line of characters in the input."""
    
    # Turn into standard format
    m = _to_array(data=data)
    m = standardize(m, pixel_on=pixel_on)

    # Load the characters and ascii art glyphs
    char_glyphs_pairs = load_glyphs(data=m, glyph_dimensions_pixels=glyph_dimensions_pixels)

    res = ""
    scanner = Scanner(data=m)

    # Scan left to right across the input, looking for matching ASCII art-like glyphs
    while not scanner.done():
        # Check for matches at the current location
        for char, glyph in char_glyphs_pairs:
            if scanner.match(glyph, skip_ahead_on_match=True):
                res += char
                break
            #
        else:
            # If no glyphs match, skip ahead to the next line
            scanner.skip_ahead()
    
    return res


_example = """
####.####.####.#...##..#.####.###..####..###...##.
#....#....#....#...##.#..#....#..#.#......#.....#.
###..###..###...#.#.##...###..#..#.###....#.....#.
#....#....#......#..#.#..#....###..#......#.....#.
#....#....#......#..#.#..#....#.#..#......#..#..#.
####.#....####...#..#..#.#....#..#.#.....###..##..
"""


if __name__ == '__main__':
    parsed = ocr(_example)
    print(parsed)

    # Make sure switching characters doesn't change the output    
    assert parsed == ocr(_example.replace("#", "@").replace(".", "x"))
