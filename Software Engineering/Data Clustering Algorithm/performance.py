import numpy as np
import timeit
import matplotlib.pyplot as plt
from clustering_numpy import cluster as cluster_with_numpy
from clustering import cluster as cluster_without_numpy


time = timeit.default_timer()

k = 3
positions = np.loadtxt('data/samples_random.csv')
time = timeit.default_timer() - time
print('Reading in data time:', time)

batch_size = [100, 500, 1000, 2000, 4000, 10000]

nump_time = np.zeros((6, 6))
no_nump_time = np.zeros((6,6))
for loop in range(6):
    for batch in range(6):
        position_data = positions[:batch_size[batch]]
        time = timeit.default_timer()
        output_nump = cluster_with_numpy(position_data, 10, k)
        nump_time[batch][loop] = timeit.default_timer() - time
        nump_time_avg = np.sum(nump_time, axis= 1)/ len(batch_size)

for loop in range(6):
    for batch in range(6):
        position_data = positions[:batch_size[batch]]
        time = timeit.default_timer()
        output_no_nump = cluster_without_numpy(position_data,10,k)
        no_nump_time[batch][loop] = timeit.default_timer() - time
        no_nump_time_avg = np.sum(no_nump_time, axis=1)/len(batch_size)

plt.plot(batch_size, no_nump_time_avg, label="non-numpy", color = "cyan")
plt.plot(batch_size, nump_time_avg, label="numpy", color= "black")
# use this for scatter
plt.scatter(batch_size, no_nump_time_avg, label="non-numpy", color="cyan", marker="x", s=60)
plt.scatter(batch_size, nump_time_avg, label="numpy", color="black", marker=",", s=60)
plt.title("Average computation time over 6 repeats")
plt.xlabel("Batch_size")
plt.ylabel("Time (sec)")
plt.legend()
plt.savefig('performance.png')
plt.show()











