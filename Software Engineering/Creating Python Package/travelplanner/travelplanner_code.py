import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


def read_passenger(file):
    # Check File Exists
    if os.path.exists(file):
        pass
    else:
        raise FileExistsError('File not found')
    # Check File Input is actually CSV
    if file.endswith('.csv'):
        pass
    else:
        raise TypeError('Please use a CSV File')
    passengers_data = []
    with open(file) as f:
        newline = csv.reader(f)
        for line in newline:
            line = [int(x) for x in line]
            passengers_data.append(((line[0], line[1]),
                                    (line[2], line[3]), line[4]
                                    ))
    return passengers_data


def read_route(route_file):
    # Check File Exists
    if os.path.exists(route_file):
        pass
    else:
        raise FileExistsError('File not found')
    # Check File is a CSV
    if route_file.endswith('.csv'):
        pass
    else:
        raise TypeError('Please use a CSV File')
    with open(route_file) as f:
        csvfile = csv.reader(f)
        route_coordinates = []
        for line in csvfile:
            x = int(line[0])
            y = int(line[1])

            if len(line) > 1:
                stop = line[2]
            else:
                stop = ''
            coordinates = (x, y, stop)
            route_coordinates.append(coordinates)
    return route_coordinates


class Passenger:
    def __init__(self, start, end, speed):
        self.start = start
        self.end = end
        self.speed = speed

        # Check starting point has x,y
        if type(start) != tuple:
            raise TypeError('Please specify starting point as a tuple')
        else:
            pass
        # Check ending point has x,y
        if type(end) != tuple:
            raise TypeError('Please specify starting point as a tuple of x,y')
        else:
            pass
        # Check speed is given as int
        if type(speed) != int:
            raise TypeError('Please specify starting point as a integer')
        else:
            pass
        # Check speed is 0
        if speed <= 0:
            raise ValueError(
                'Please specify a positive speed ')
        assert (len(start) == 2), \
            'Please ensure starting location Tuple has only two elements'
        assert (len(end) == 2), \
            "Please ensure ending location Tuple has only two elements"

    def walk_time(self):
        """Calculates the time taken to walk from start -> finish"""
        distance = [(math.sqrt((self.end[0] - self.start[0]) ** 2 +
                               (self.end[1] - self.start[1]) ** 2))]
        return distance * self.speed


class Route(list):
    def __init__(self, filename):
        super().__init__()
        self.extend(read_route(filename))
        # Check for diagonal routes
        for k in range(len(self) - 1):
            if self[k][0] != self[k + 1][0] and self[k][1] != self[k + 1][1]:
                raise ValueError('The bus cannot move diagonally')

    def plot_map(self, saveplots):
        """Plots a savable map of the route"""
        max_x = max([n[0] for n in self]) + 5  # adds padding

        max_y = max([n[1] for n in self]) + 5
        grid = np.zeros((max_y, max_x))
        for x, y, stop in self:
            grid[y, x] = 1
        if stop:
            grid[y, x] += 1
        fig, ax = plt.subplots(1, 1)
        ax.pcolor(grid)
        ax.invert_yaxis()
        ax.set_aspect('equal', 'datalim')
        plt.show()
        if saveplots:
            print('Saving Plot as map.png')
            plt.savefig('map.png')
        else:
            pass

    def timetable(self, speed=10):
        """
        Generates a timetable for a route as minutes from its first stop.
        """
        if speed < 0:
            raise ValueError('Please input a positive speed')
        time = 0
        stops = {}
        for step in self:
            if step[2]:
                stops[step[2]] = time
            time += speed
        return stops

    def generate_cc(self):
        """
        Converts a set of route into a Freeman chain code
        3 2 1
         |
        4 - C - 0
        / | \
        5 6 7
        """

        start = self[0][:2]
        cc = []
        freeman_cc2coord = {0: (1, 0),
                            1: (1, -1),
                            2: (0, -1),
                            3: (-1, -1),
                            4: (-1, 0),
                            5: (-1, 1),
                            6: (0, 1),
                            7: (1, 1)}
        freeman_coord2cc = {val: key for key, val in freeman_cc2coord.items()}

        for b, a in zip(self[1:], self):
            x_step = b[0] - a[0]
            y_step = b[1] - a[1]
            cc.append(str(freeman_coord2cc[(x_step, y_step)]))
        return start, ''.join(cc)

    # def allowedroute(self):
    #
    #     startpoint, diagcheck = self.generate_cc()
    #     diagcheck = list(diagcheck)
    #     istep = 0
    #     for check in diagcheck:
    #         istep += 1
    #         if int(check) % 2 == 1:
    #             print('Allowed Route')
    #         else:
    #             print('Diagonal Routes are not accepted')


class Journey(object):
    def __init__(self, route, passengers):
        self.route = route
        self.passengers = passengers

    def plot_bus_load(self, saveplots):
        stops = {step[2]: 0 for step in self.route if step[2]}

        for passenger in self.passengers:
            trip = passenger_trip(passenger, self.route)
        stops[trip[0][1]] += 1
        stops[trip[1][1]] -= 1
        for i, stop in enumerate(stops):
            if i > 0:
                stops[stop] += stops[prev]
            prev = stop
        fig, ax = plt.subplots()
        ax.step(range(len(stops)), list(stops.values()), where='post')
        ax.set_xticks(range(len(stops)))
        ax.set_xticklabels(list(stops.keys()))
        plt.show()
        if saveplots:
            print('Saving plot as load.png')
            plt.savefig('load.png')
        else:
            pass

    def travel_time(self, passenger_id, speed=10):
        """Calculates the total travel time"""
        passenger = self.passengers[passenger_id]
        closer_start, closer_end = passenger_trip(passenger, self.route)
        bus_timetable = self.route.timetable(speed)

        # Make sure passengers only get on the bus in a forward direction
        if closer_end[1] <= closer_start[1]:
            bus_travel = 0
            walk_travel = self.passengers[passenger_id].speed * \
                (math.sqrt((passenger.end[0] -
                 passenger.start[0]) ** 2 + (passenger.end[1] -
                 passenger.start[1]) ** 2))
        else:
            bus_travel = bus_timetable[closer_end[1]] - \
                         bus_timetable[closer_start[1]]

            walk_travel = closer_start[0] * self.passengers[passenger_id].speed \
                + closer_end[0] * self.passengers[passenger_id].speed
            walk_travel = round(walk_travel, 1)

        return {'bus': bus_travel, 'walk': walk_travel}

    def print_time_stats(self):
        """Print everything about the journey"""
        for traveller in self.passengers:
            trip = self.passenger_trip(passenger=traveller)
            walk_time = traveller.walk_time()
            bus_times = Route.timetable(self.route)
            bus_travel = bus_times[trip[1][1]] - \
                bus_times[trip[0][1]]
            walk_travel = trip[0][0] * traveller.speed + \
                trip[1][0] * traveller.speed
            if bus_travel + walk_travel > walk_time:
                self.total_walk += walk_time
            else:
                self.total_bus += bus_travel
                self.total_walk += walk_travel

            passenger_number = len(self.passengers)
            print('Average time on bus:%2f min'
                  'Average walking time: %2f: min' %
                  self.total_bus / passenger_number,
                  self.total_walk / passenger_number)


def passenger_trip(passenger, route):
    """
        Returns the best route to take for your journey, for example
        --------
        >>> route = Route("travelplanner/tests/route.csv")
        >>> passenger_trip(Passenger(start=(6, 12), end=(16, 0), speed=5), route)
        ((9.486832980505138, 'A'), (6.082762530298219, 'B'))
        """
    assert (isinstance(passenger, Passenger)), (
        "passenger input not Passenger type")
    assert (isinstance(route, Route)), (
        "route input not Route type")
    start, end, pace = passenger.start, passenger.end, passenger.speed
    stops = [value for value in route if value[2]]
    # calculate closer stops
    # to start
    distances = [(math.sqrt((x - start[0]) ** 2 +
                 (y - start[1]) ** 2), stop) for x, y, stop in stops]
    closer_start = min(distances)
    # to end
    distances = [(math.sqrt((x - end[0]) ** 2 +
                            (y - end[1]) ** 2), stop) for x, y, stop in stops]
    closer_end = min(distances)
    return closer_start, closer_end
