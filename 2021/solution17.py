import numpy as np

# Read in the data
with open("input17.txt") as f:
    # target area: x=269..292, y=-68..-44
    stuff = f.read().strip().split("target area: ")[1].split(", ")
    borders = []
    for s in stuff:
        border = tuple(int(x) for x in s.split("=")[1].split(".."))
        borders.append(border)

# Parse the x and y borders of the target area we'll have to hit with the probe
xlim, ylim = borders


def within_borders(coords, borders):
    """Determines whether input coordinates fall within target area"""
    for coord, (a, b) in zip(coords, borders):
        if not (a <= coord <= b):
            return False
        #
    return True


def missed(coords, borders):
    """Determines if a shot has missed the target. It has missed if it's too far right or down,
    because the probe can never go left or up."""
    return coords[0] > borders[0][1] or coords[1] < borders[1][0]


class Probe:
    def __init__(self, pos, vel):
        self.pos = np.array(pos)
        self.vel = np.array(vel)

    def tick(self):
        """Updates the probe's position and velocity vectors."""
        self.pos += self.vel
        a, b = self.vel
        # y-velocity decrements by one because gravity
        b -= 1
        # x-velocity approaches 0 because friction/resistance
        if a != 0:
            a -= int(a > 0)
        self.vel = np.array([a, b])


def determine_final_delta_x(dx):
    """The final horizontal distance is x + (x - 1) + ... = x(x+1)/2 (the Gauss trick)"""
    return dx*(dx + 1)/2


def determine_min_vx(target):
    """Som initial x-velocities will never reach the target.
    This finds the minimum x-velocity such that the probe doesn't stall
    before reaching target."""
    final = float("-inf")
    running = 0
    while final < target:
        running += 1
        final = determine_final_delta_x(running)
    return running


def determine_max_vy(target):
    """After the probe comes back down to y=0, if the vertical speed is so great the probe
    passes through the target, we've exceeded the max y-velocity."""
    vy = -target
    return vy

# Determine the parameter space to search
min_vx = determine_min_vx(target=xlim[0])
max_vx = xlim[1] + 1
min_vy = ylim[0] - 1
max_vy = determine_max_vy(ylim[0])
pos = [0,0]

highest_y = float('-inf')
n_hits = 0
for vx in range(min_vx, max_vx+1):
    for vy in range(min_vy, max_vy+1):
        thisymax = float("-inf")
        probe = Probe(pos=pos, vel=[vx, vy])
        # While probe isn't in the target, update its position
        while not within_borders(probe.pos, borders):
            probe.tick()
            thisymax = max(thisymax, probe.pos[1])
            # If it passed the target, we're done
            if missed(probe.pos, borders):
                break
            #
        hit = within_borders(probe.pos, borders)
        if hit:
            highest_y = max(highest_y, thisymax)
            n_hits += 1

print(f"Solution to star 1: {highest_y}.")
print(f"Solution to star 2: {n_hits}.")