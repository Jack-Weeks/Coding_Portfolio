import unittest
from travelplanner.travelplanner_code import read_passenger, read_route


class Testing_Read_Passenger(unittest.TestCase):
    def test_string_input_pass(self):
        self.assertRaises(FileExistsError,
                          lambda: read_passenger(
                              'travelplanner/passengers.csv'))
        self.assertRaises(TypeError,
                          lambda: read_passenger(
                              'travelplanner/tests/test_csv_reader.py'))

    def test_string_input_route(self):
        self.assertRaises(FileExistsError,
                          lambda: read_route(
                              'travelplanner/route.csv'))
        self.assertRaises(TypeError,
                          lambda: read_route(
                              'travelplanner/tests/test_csv_reader.py'))
