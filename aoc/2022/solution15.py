import re


def read_input():
    with open("input15.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    sensors = []
    beacons = []

    for line in s.split("\n"):
        match = re.match(r'Sensor at x=(.*), y=(.*): closest beacon is at x=(.*), y=(.*)', line)
        sx, sy, bx, by = map(int, match.groups())
        sensors.append((sx, sy))
        beacons.append((bx, by))

    return sensors, beacons


def manhatten_distance(a, b):
    """Returns the Manhatten distance between points a and b, both specified as (x, y)."""
    res = 0
    for d1, d2 in zip(a, b):
        res += abs(d1 - d2)

    return res


def combine_spans(spans):
    """Combined an iterable of span tuples into a more dense representation of the interval they span.
    For instance [(1, 3), (4, 7)] results in [(1, 7)].
    Works by starting with the span with the lowest lower edge, then repeatedly expands to include other spans."""

    ordered = sorted([span for span in spans if span is not None], key=lambda t: t[0])  # sort tuples by their left edge
    res = []
    running = None

    for span in ordered:
        # Start with the first span
        if running is None:
            running = span
            continue

        running_left, running_right = running
        span_left, span_right = span

        connected = running_right + 1 >= span_left
        if connected:
            # If the intervals overlap, expand running interval to include the newly encountered span
            running = (running_left, max(span_right, running_right))
        else:
            # Otherwise, append to results and start with a new running interval
            res.append(running)
            running = span
        #
    if running:
        res.append(running)

    return res


def exclusion_from_y_single_sensor(sensor, beacon, y):
    """Takes a sensor and a beacon tuple ((x, y) format), and a y-value.
    Returns the span of x-values we can exclude based on the given sensor and beacon.
    For example, if the sensor+beacon exclude x-values 7,8,9, this method returns (7, 9)."""

    dist = manhatten_distance(sensor, beacon)
    sensor_x, sensor_y = sensor

    dist_y = abs(sensor_y - y)
    max_dist_x = abs(dist - dist_y)

    if dist_y > dist:
        return None

    span = (sensor_x - max_dist_x, sensor_x + max_dist_x)

    return span


def exclusion_from_y_multiple_sensors(sensors, beacons, y):
    """Takes lists of sensors and beacons, and a value of y, and returns the span(s) of x-values
    in which we can exclude beacons."""

    # Get the exclusion spans for each sensor-beacon pair
    exclusion_spans = []
    for sensor, beacon in zip(sensors, beacons):
        span = exclusion_from_y_single_sensor(sensor, beacon, y)
        exclusion_spans.append(span)

    # Combine them and return
    exclusion = combine_spans(exclusion_spans)
    return exclusion


def count_excluded_x_values(sensors, beacons, y):
    """Returns the number of x-values where we can exclude a beacon being present, based on sensors+beacons."""

    # Get the excluded span(s) from sensor-beacon data.
    exclusion = exclusion_from_y_multiple_sensors(sensors, beacons, y)
    res = sum(b + 1 - a for a, b in exclusion)

    # Some x-values might be occupied by beacons. These should not be counted.
    beacons_on_line_xvals = {bx for bx, by in beacons if by == y}
    for bx in beacons_on_line_xvals:
        res -= any(a <= bx <= b for a, b in exclusion)

    return res


def find_single_excluded_coords(sensors, rows, cut):
    """Finds the x and y coords which is the only point at which a beacon can exist and be missed by the sensors."""

    max_msg_len = 0
    for y in range(cut+1):
        exclusion = exclusion_from_y_multiple_sensors(sensors, rows, y)
        if len(exclusion) > 1:
            a = exclusion[0][1]
            b = exclusion[1][0]

            # The space between the excluded regions should only be a single cell
            assert b - a == 2
            x = a + 1

            if all(0 <= val <= cut for val in (x, y)):
                print(" "*max_msg_len, end="\r")
                return x, y
            #
        if y % 10000 == 0:
            msg = f"Checked {y} y-values ({100*(y+1)/cut:.1f}%)."
            print(msg, end="\r")
            max_msg_len = max(max_msg_len, len(msg))


def determine_tuning_frequency(sensors, beacons, cut):
    x, y = find_single_excluded_coords(sensors, beacons, cut)
    res = 4000000*x + y
    return res


def main():
    raw = read_input()
    sensors, beacons = parse(raw)

    row = 2000000
    n_excluded = count_excluded_x_values(sensors, beacons, y=row)
    print(f"At row {row}, there are {n_excluded} cells where a beacon can't exist.")

    tuning_freq = determine_tuning_frequency(sensors, beacons, cut=4000000)
    print(f"Tuning frequency is: {tuning_freq}.")


if __name__ == '__main__':
    main()
