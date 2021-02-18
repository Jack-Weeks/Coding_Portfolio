from travelplanner.travelplanner_code import *


def bussimula(passfile, routefile, speed=10, saveplots=False):
    route = Route(routefile)
    passengerdata = read_passenger(passfile)
    passengers = []
    for i in range(len(passengerdata)):
        passenger = (
            Passenger(start=passengerdata[i][0], end=passengerdata[i][1],
                      speed=passengerdata[i][2]
                      ))
        passengers.append(passenger)

    journey = Journey(route, passengers)

    print(" Stops: minutes from start\n", Route.timetable(route))

    for passenger_id in range(len(passengers)):
        print(f"Trip for passenger: {passenger_id}")
        start, end = passenger_trip(passenger, route)
        total_time_d = journey.travel_time(passenger_id, int(speed))
        total_time = total_time_d['bus'] + total_time_d['walk']

        print((f" Walk {start[0]:3.2f} units to stop {start[1]}, \n"
               f" get on the bus and alite at stop {end[1]} and \n"
               f" walk {end[0]:3.2f} units to your destination."))
        print(f" Total time of travel: {total_time:03.2f} minutes")

    if saveplots:
        # Saves the route.
        route.plot_map(saveplots)
        # Saves the bus load as a function of time.
        journey.plot_bus_load(saveplots)
