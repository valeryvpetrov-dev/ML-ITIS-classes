import matplotlib.pyplot as plt
import numpy as np
from enum import Enum


class Flag(Enum):
    UNKNOWN = -1
    GREEN = 0
    YELLOW = 1
    RED = 2


def euclidean_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


if __name__ == '__main__':
    n = 100
    x = np.random.randint(1, 100, n)
    y = np.random.randint(1, 100, n)
    eps, min_neighbours = 10, 3
    flags = [Flag.UNKNOWN.value for i in range(n)]

    # assign green flags
    for i in range(n):
        neighbours = 0
        for j in range(n):
            if i == j:
                continue
            if euclidean_dist(x[i], y[i], x[j], y[j]) < eps:
                neighbours += 1
        if neighbours >= min_neighbours:
            flags[i] = Flag.GREEN.value

    # assign yellow and red flags
    for i in range(n):
        if flags[i] == Flag.UNKNOWN.value:
            for j in range(n):
                if flags[j] == Flag.GREEN.value and euclidean_dist(x[i], y[i], x[j], y[j]) < eps:
                    flags[i] = Flag.YELLOW.value
                    break
        if flags[i] == Flag.UNKNOWN.value:
            flags[i] = Flag.RED.value

    # assign points to clusters
    clusters = [0 for i in range(n)]
    last_cluster_marker = 1
    for i in range(n):
        if flags[i] == Flag.GREEN.value:
            if clusters[i] == 0:
                clusters[i] = last_cluster_marker
                last_cluster_marker += 1
            for j in range(n):
                if euclidean_dist(x[i], y[i], x[j], y[j]) < eps:
                    # TODO now it assign to last cluster
                    clusters[j] = clusters[i]

    # TODO show unsigned points
    # plot clustered points
    for i in range(n):
        cluster_color = clusters[i] / last_cluster_marker
        if clusters[i] == Flag.UNKNOWN.value:
            plt.scatter(x[i], y[i], color='black', marker='*')
        else:
            plt.scatter(x[i], y[i], color=(cluster_color, 0.2, cluster_color ** 2))
    plt.show()
