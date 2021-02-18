from argparse import ArgumentParser
from travelplanner.output import bussimula


def travel_planner_input():
    parser = ArgumentParser(
        description='Plan your best route for your journey')
    parser.add_argument('passfile',
                        help='Input a file path relating to a CSV file '
                             'containing passenger data')
    parser.add_argument('routefile',
                        help='Input a file path relating to a CSV file '
                             'containing passenger route data')
    parser.add_argument('--speed', default=10,
                        help='Input the bus speed, as an integer value'
                             'default is 10')
    parser.add_argument('--saveplots', action='store_true',
                        help='Gives the option whether you would like'
                             'to save the output route.')
    arguments = parser.parse_args()

    print(bussimula(arguments.routefile, arguments.passfile,
                    arguments.speed, arguments.saveplots))


if __name__ == '__main__':
    travel_planner_input()
