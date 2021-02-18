import numpy as np
import argparse


def main():
    k = 3
    positions = np.loadtxt(arguments.datafile_path, delimiter=',')

    cluster(positions, arguments.iters, k)


def cluster(positions, num_iters, k):
    """
    Calculates the cluster centres for k clusters and the number of points
     in each cluster, classifies each datapoint to one of the clusters.
    :param positions: (tuple) list of position coordinates.
    :param num_iters: (int) number of iterations the cluster algorithm runs.
    :param k: (int) the number of clusters points are classified into.
    :return:
    """
    cluster_center = initalise_cluster(positions, k)
    n = 0
    while n < num_iters:
        closest_cluster = get_distances(positions, cluster_center)
        cluster_center, cluster_size = updated_centres(positions, k, closest_cluster)
        n += 1
    for i in range(k):
        print("Cluster " + str(i)
              + " is centred at " + str(cluster_center[i])
              + " and has " + str(cluster_size[i]) + " points.")


def initalise_cluster(positions, k):
    """
    Randomly generates the positions of the clusters.
    :param positions: list of (tuple): list of position coordinates.
    :param k: (int the number of clusters points are classified into.
    :return: cluster_centre: list of (tuple) lenght k of centroid coordinates
    """
    cluster_centre = positions[np.random.randint(positions.shape[0], size=k)]
    return cluster_centre


def get_distances(positions, cluster_center):
    """
    Calculates the euclidean distance between the point and the current cluster centre

    :param positions: list of (tuple) containing position coordinates.
    :param cluster_center:  list of (tuple) containing of centroid coordinates.
    :return: closest_cluster: (tuple) of the coordinates of the closest cluster
    """
    broadcasting_axis = cluster_center[:, np.newaxis, :]
    distances = np.sum(np.square(positions - broadcasting_axis), axis=2)
    distances = np.sqrt(distances)
    closest_cluster = np.argmin(distances, axis=0)
    return closest_cluster


def updated_centres(positions, k, closest_cluster):
    """
    Calculates the new cluster centre as a function of the current points assigned to the cluster

    :param positions:
    :param k: (int the number of clusters points are classified into.
    :param closest_cluster: (tuple) of the coordinates of the closest cluster
    :return: cluster_centers: list of len k of (tuples)
             cluster_size: (int) number of points in the cluster.
    """
    regionsidx = np.array([0, 1, 2])
    regionsidx = regionsidx.reshape(1, -1)
    closest_cluster = np.array(closest_cluster)
    closest_cluster = closest_cluster.reshape(-1, 1)
    find_closest = regionsidx == closest_cluster
    find_closest = find_closest.astype(int)
    cluster_size = np.sum(find_closest, axis=0)
    allocate_cluster = positions[:, :, np.newaxis] * find_closest[:, np.newaxis, :]
    new_centers = np.sum(allocate_cluster, axis=0) / cluster_size
    cluster_centers = np.transpose(new_centers)
    return cluster_centers, cluster_size



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='K means clustering command line interface')
    parser.add_argument('--datafile_path', default='data/samples.csv', type=str,
                        help='Add the filepath to your datafile you wish to perform the clustering aglorithm upon.')
    parser.add_argument('--iters', default=10, type=int,
                        help='State how many iterations you wish to perform the algorithm over, default is 10')
    arguments = parser.parse_args()

    main()