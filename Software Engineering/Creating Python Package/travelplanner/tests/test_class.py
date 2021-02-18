import unittest
from travelplanner.travelplanner_code import Passenger, Route, \
    Journey, passenger_trip
from pytest import raises


class test_Passenger_Class(unittest.TestCase):
    def test_Passenger_Constructor(self):
        passenger = Passenger(start=(6, 6), end=(6, 12), speed=4)
        self.assertEqual(passenger.start, (6, 6))
        self.assertEqual(passenger.end, (6, 12))
        self.assertEqual(passenger.speed, 4)

    def test_passenger_input(self):
        self.assertRaises(TypeError,
                          lambda:
                          Passenger(start=10, end=(10, 12), speed=6))
        self.assertRaises(TypeError,
                          lambda:
                          Passenger(start=(6, 12), end=12, speed=6))
        self.assertRaises(TypeError,
                          lambda:
                          Passenger(start=4, end=(10, 12), speed=3.9))
        self.assertRaises(ValueError,
                          lambda:
                          Passenger(start=(6, 12), end=(10, 12), speed=-2))
        self.assertRaises(AssertionError,
                          lambda:
                          Passenger(start=(6, 12, 3), end=(10, 12), speed=6))
        self.assertRaises(AssertionError,
                          lambda:
                          Passenger(start=(6, 12), end=(9, 12, 3), speed=3))


class test_route_class(unittest.TestCase):
    def test_route_constructor(self):
        route = Route('travelplanner/tests/route.csv')
        actual_route = [
            (15, 9, 'A'), (15, 8, ''), (15, 7, ''), (15, 6, 'B')]
        self.assertEqual(route, actual_route)

    def test_raises_route_errors(self):
        self.assertRaises(ValueError,
                          lambda: Route('travelplanner/tests/diagroute.csv'))

    def test_timetable(self):
        route = Route('travelplanner/tests/route.csv')
        self.assertRaises(ValueError,
                          lambda: Route.timetable(route, -5))
        self.assertEqual(
            Route.timetable(route), {'A': 0, 'B': 30})
        self.assertEqual(
            Route.timetable(route, 15), {'A': 0, 'B': 45})

    def test_route_cc(self):
        route = Route('travelplanner/tests/route.csv')
        self.assertEqual(route.generate_cc(), ((15, 9), '222'))


class test_journey_class(unittest.TestCase):
    def test_journey_constructor(self):
        john = Passenger(start=(6, 12), end=(10, 12), speed=5)
        mary = Passenger(start=(0, 0), end=(6, 2), speed=12)
        route = Route('travelplanner/tests/route.csv')
        journey = Journey(route, [john, mary])
        self.assertEqual(journey.route, route)
        self.assertEqual(journey.passengers, [john, mary])

    def test_journey_input(self):
        pass

    # def test_print_time_stats(self):
    #     john = Passenger(start=(6, 12), end=(10, 12), speed=5)
    #     mary = Passenger(start=(0, 0), end=(6, 2), speed=12)
    #     route = Route('travelplanner/tests/route.csv')
    #     journey = Journey(route, [john, mary])

    def test_travel_time(self):
        john = Passenger(start=(6, 12), end=(10, 12), speed=5)
        mary = Passenger(start=(0, 0), end=(6, 2), speed=12)
        route = Route('travelplanner/tests/route.csv')
        journey = Journey(route, [john, mary])
        bus_time = journey.travel_time(0)['bus']
        walk_time = journey.travel_time(0)['walk']
        self.assertEqual(bus_time, 0)
        self.assertEqual(walk_time, 20)


class test_external_functions(unittest.TestCase):
    def test_passenger_trip(self):
        passenger = Passenger(start=(6, 12), end=(10, 12), speed=5)
        route = Route('travelplanner/tests/route.csv')
        self.assertRaises(AssertionError,
                          lambda: passenger_trip(passenger, passenger))
        self.assertRaises(AssertionError,
                          lambda: passenger_trip(route, route))


def pytest_raises():
    with raises(TypeError) as exception:
        Passenger(start=4, end=(10, 12), speed=5)
