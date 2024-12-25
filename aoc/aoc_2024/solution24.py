# ꞏ•  ꞏ⸳ •.    ꞏ⸳  •.  ꞏ⸳ `   +ꞏ   .*   ⸳ꞏ*`⸳       ꞏ.`+⸳ꞏ   .*  •   ⸳ ꞏ ⸳ꞏ*⸳.  
# ` ꞏ⸳  `. .+   ꞏ  ⸳`*ꞏ⸳•`  ꞏ.⸳   Crossed Wires `ꞏ  `ꞏ⸳      ` * ꞏ⸳ •  `   ⸳.•`ꞏ
# ⸳  ꞏ• .`ꞏ ⸳   .*     https://adventofcode.com/2024/day/24 * ꞏ.   ⸳  .  `+ ⸳+ `
# .  ` ꞏ  •*`.ꞏ⸳  .`   +⸳ .    .⸳ .  ꞏ  ꞏ `*    ⸳ ꞏ.   ⸳*ꞏ ꞏ`  +ꞏ⸳.    * ꞏ  `ꞏ..


def parse(s):
    """Parses input into 2 dicts - one mapping variables to their starting value, another mapping variables
    to a tuple like (a, operator, b), indicating how the variable's value is computed from 2 inputs,
    e.g. ('x', "AND", "y")"""

    valpart, rulepart = s.split("\n\n")
    
    initial_values = {a: int(b) for a, b in map(lambda line: line.split(": "), valpart.splitlines())}
    
    rules = dict()
    for line in rulepart.splitlines():
        inputpart, target = line.split(" -> ")
        a, operator, b = inputpart.split()
        rules[target] = (a, operator, b)

    return initial_values, rules


class MonitoringDevice:
    """Represents the monitoring device. Handles stuff like recursively resolving variable values
    in terms of their inputs, looking up formulas for variables, and swapping inputs."""
    
    def __init__(self, initial_values: dict, rules: dict):
        # Store both variable -> formula, and the inverse, so both are straightforward to look up
        self.rules = {k: v for k, v in rules.items()}
        self.rules_flipped = {v: k for k, v in rules.items()}
        assert len(self.rules) == len(self.rules_flipped)
        
        # Set the starting values
        self.values = dict()
        self.set_values(**initial_values)
        
        # Keep track of any inputs that have been swapped
        self._swapped = []

    
    def set_values(self, clear=True, **kwargs):
        """Sets the specified key/value pairs. If clear is True, deletes all values first."""
        if clear:
            self.values = dict()
        for k, v in kwargs.items():
            self.values[k] = v
        #

    def swap_inputs(self, replace: dict):
        """Swaps two inputs. replace is a dict mapping values to each other, e.g.
        {a: b, b: a}"""

        # Only allow swapping a single pair at a time
        assert len(replace) == 2
        self._swapped.append(tuple(replace.keys()))
        
        # Update all formulas involving the swapped wires
        for output, formula in self.rules.items():
            if not any(var in replace for var in formula):
                continue  # Don't bother with other wires
            
            # Update the rules (output -> formula)
            rewired = tuple(replace.get(old, old) for old in formula)
            self.rules[output] = rewired
            
            # Update the inverse rules (output -> target)
            del self.rules_flipped[formula]
            self.rules_flipped[rewired] = output
        
        assert len(self.rules) == len(self.rules_flipped)
    
    def _do_operation(self, a: int, operation: str, b: int):
        """Runs a fundamental logical operation on two integer inputs."""
        
        # Double-check the input is binary
        if not all(val in (0, 1) for val in (a, b)):
            raise ValueError(f"Integer arguments must be binary - got: {(a, b)}")
        
        # Look up the correct operation and to the Boolean algebra
        res = None
        match operation:
            case "AND":
                res = a*b
            case "OR":
                res = a + b - a*b
            case "XOR":
                res = a + b - 2*a*b
            case _:
                raise ValueError(f"Got invalid operation: {operation}.")
            #
        return res
    
    def resolve(self, wire: str):
        """Resolves the value of the specified wire and store it in the monitor's values dictionary.
        Any yet unresolved inputs encountered, are recursively resolved and saved as well."""
        
        if wire in self.values:
            return  # Already got the value, nothing to do
        
        # Determine the formula for the desired value
        formula = self.rules[wire]
        a, op, b = formula
        # Make sure the components of the formula are resolved
        for arg in (a, b):
            self.resolve(arg)
        
        # After resolving, look up the components and compute the resulting values
        val_a, val_b = (self.values[arg] for arg in (a, b))
        res = self._do_operation(val_a, op, val_b)
        self.values[wire] = res
    
    def resolve_all(self):
        """Resolves all wire values"""
        for wire in self.rules.keys():
            self.resolve(wire=wire)
        #
    
    def get_number(self, prefix="z"):
        """Takes the bit values of the bits whose variables start with the specified prefix.
        Returns the integer which the bits represent."""
        keys = sorted(k for k in self.values.keys() if k.startswith(prefix))
        res = sum(self.values[k]*2**i for i, k in enumerate(keys))
        return res
    
    def _lookup(self, a, op, b):
        """Looks up rules where the specified operation is run on the specified inputs.
        All logical operations in the problem are symmetric, so we try looking for both permutations
        of the two values to make sure we don't miss it.
        Returns None if the formula doesn't exist in the rules."""

        try:
            k = next((key for key in ((a, op, b), (b, op, a)) if key in self.rules_flipped))
            return self.rules_flipped[k]
        except StopIteration as e:
            return None
        #

    def _discover_rewires(self, a, op, b) -> dict:
        """Looks for existing rules which can be reqired into the input rule by swapping two inputs.
        Returns a dict representing the swap, e.g. {a: b, b: a}"""
        
        # Find all rules where a single swapping of inputs will result in the input formula
        possible_swaps = []
        for ka, kop, kb in self.rules_flipped.keys():
            if kop != op:
                continue  # Skip formulas with the wrong logical operation
            
            # Skip formulas where more than one replacement is required to produce the target formula
            replacements_needed = sum(var not in (ka, kb) for var in (a, b))
            if replacements_needed != 1:
                continue
            
            possible_swaps.append(tuple(sorted({a, b} ^ {ka, kb})))
        
        # Assume unambiguity (handle multiple options if this throws an error)
        assert len(possible_swaps) == 1
        swap_wires = possible_swaps[0]
        
        old, new = swap_wires
        res = {old: new, new: old}
        return res

    def lookup_or_rewire(self, a, op, b):
        """Attempt to look up the formula (a, <operation>, b).
        If it already exists, return it. If not, identify the necessary input swap to create it, then return it."""
        
        # Try normal lookup
        res = self._lookup(a, op, b)
        if res is not None:
            return res

        # If missing, swap wires to make it e
        swap = self._discover_rewires(a, op, b)
        self.swap_inputs(swap)
        res = self._lookup(a, op, b)
        
        # It shouldn't be possible for it to still be missing now, but check it just in case
        if res is None:
            raise RuntimeError(f"After attempting swap, still couldn't find {(a, op, b)}")
        
        return res 
    
        
    def trace_binary_addition(self, xvar="x", yvar="y"):
        """Trace through a binary addition of two variables.
        At each step, the required logical operations are determined and looked up in the rules.
        When rules are missing, the required wire swaps are determined and implemented on the fly."""
        
        # Get the bit symbols for the two inputs
        xbits, ybits = (sorted((k for k in self.values.keys() if k.startswith(char)), key=str) for char in (xvar, yvar))
        and_, or_, xor_ = ("AND", "OR", "XOR")
        
        carry = None
    
        for x, y in zip(xbits, ybits):
            # First half-add
            sum1 = self._lookup(x, xor_, y)
            carry1 = self._lookup(x, and_, y)
            
            # For the first iteration, nothing more to do
            if carry is None:
                carry = carry1
                continue
            
            # Second half-adder - compute result bit value and carry
            _ = self.lookup_or_rewire(sum1, xor_, carry)
            carry2 = self.lookup_or_rewire(sum1, and_, carry)
            carry = self.lookup_or_rewire(carry1, or_, carry2)
        #
    #


def identify_swaps(initial_values: dict, rules: dict):
    """Identify the required swaps for the monitor thingy to output the sum of z and y."""
    
    mon = MonitoringDevice(initial_values=initial_values, rules=rules)
    mon.trace_binary_addition()
    
    return mon._swapped
    

def solve(data: str):
    initial_values, rules = parse(data)
    
    monitor = MonitoringDevice(initial_values=initial_values, rules=rules)
    monitor.resolve_all()
    star1 = monitor.get_number()
    print(f"Solution to part 1: {star1}")

    swaps_needed = identify_swaps(initial_values=initial_values, rules=rules)
    swapped_wires = sum(map(list, swaps_needed), [])
    star2 = ",".join(sorted(swapped_wires))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2024, 24
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
