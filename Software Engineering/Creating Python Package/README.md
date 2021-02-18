#TravelPlanner README

The travelplanner program is designed to allow you to select the most optimal route for your specified journey,
all you have to do it, generate a csv for your route, and your approximate speed of walking into
separate csv, with your start and end point and invoke:

bussimula 'your_route_csv' 'number_of_passengers_csv' (optional: --bus_speed --saveplots)

bus_speed must be given as a positive integer!

#Classes
this package contains 3 classes:
__Passenger__: Takes a starting tuple, finishing tuple and a walking speed for the passenger.
The class also contains the walk_time function which simply calculates the time it would take to walk from
start to finish along a given route.

__Route__: The route class takes a filename for a given route CSV.
The class contains the functions:

plot_map, simply plots the route on a graph!
timetable, shows the bus shedule for each stop along the route!
generate_CC, generates a route into a freeman chain code.

__Journey__: Takes the route and passenger as inputs and contains the functions:
plot_bus_load, which plots the bus population along the route, allowing you to avoid busy periods
travel_time, outputs the total travel time across bus and walking for the most optimal route
print_time_stats, gives an output of the time taken stats.

Outside of the class we have the passenger_trip function which returns the most optimal route for each
passenger to get from a->b

please refer to the CITATION and LICENCE if you would like to adapt and improve this code




 