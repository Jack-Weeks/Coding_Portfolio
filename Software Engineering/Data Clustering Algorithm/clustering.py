from math import sqrt
from random import randrange
import argparse


def calculate_distance(x1, x2, y1, y2):
    """ Calculates the euclidean distance between points.

    Parameters:
        x1 (int): x coordinate of point
        x2 (int): x coordinate of centroid centre
        y1 (int): y coordinate of point
        y2 (int): y coordinate of centroid point

        Returns:
            float: calculated distance
    """

    distance_calc = sqrt((x1 - x2) ** 2 + (y2 - y1) ** 2)
    return distance_calc


def main():
    Data_file = open(arguments.datafile_path, 'r').readlines()
    positions = []
    k = 3

    for Data_Point in Data_file:
        positions.append(tuple(map(float,
                                   Data_Point.strip().split(','))))
    cluster(positions, arguments.iters, k)


def cluster(positions, num_iters, k):
    """
    Calculates the centrepoints of k clusters and assigns each datapoint a classification.
    :param positions: (tuple): list of position coordinates.
    :param num_iters: (int): number of iterations the cluster algorithm runs.
    :param k: (int): the number of clusters points are classified into.
    :return:

    """
    len_data = len(positions)

    # Allocate random cluster starting coordinates

    cluster_centre = [positions[randrange(len_data)], positions[randrange(len_data)],
                      positions[randrange(len_data)]]
    allocated_class = []
    cluster_size = [None] * len_data
    n = 0
    while n < num_iters:
        allocated_class.clear()
        for point in positions:
            distance = [None] * k
            l = 0
            while l < k:
                # Calculate euclidean distance between the current centre and each point in the cluster
                distance[l] = calculate_distance(point[0], cluster_centre[l][0], point[1], cluster_centre[l][1])
                l += 1
            allocated_class.append(distance.index(min(distance)))

        m = 0
        while m < k:
            points_in_cluster = [data for jpoint, data in enumerate(positions)
                                 if allocated_class[jpoint] == m]
            cluster_size[m] = len(points_in_cluster)
            if cluster_size[m] <= 0:
                print('Cluster size is 0')
            # Calculate new average centrepoint for cluster in x and y
            new_mean = (
                sum([a[0] for a in points_in_cluster]) / cluster_size[m],
                sum([a[1] for a in points_in_cluster]) / cluster_size[m])
            cluster_centre[m] = new_mean
            m += 1
        n += 1
    num = 0
    while num < k:
        print("Cluster " + str(num) + " is centred at " + str(cluster_centre[num]) + " and has " +
              str(cluster_size[num]) + " points.")
        num += 1


# Argparser script

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='K means clustering command line interface')
    parser.add_argument('--datafile_path', default='data/samples.csv', type=str,
                        help='Add the filepath to your datafile you wish to perform the clustering aglorithm upon.')
    parser.add_argument('--iters', default=10, type=int,
                        help='State how many iterations you wish to perform the algorithm over, default is 10')
    arguments = parser.parse_args()

    main()
