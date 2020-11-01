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
    eps, min_neighbours = 10, 5
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

    # plot flagged points
    plt.figure(1)
    for i in range(n):
        color = None
        if flags[i] == Flag.GREEN.value:
            color = 'g'
        elif flags[i] == Flag.YELLOW.value:
            color = 'y'
        elif flags[i] == Flag.RED.value:
            color = 'r'
        plt.scatter(x[i], y[i], color=color)

    # assign points to clusters
    clusters = [-1 for i in range(n)]
    last_cluster_marker = 1
    current_cluster_marker = last_cluster_marker
    for i in range(n):
        if flags[i] == Flag.GREEN.value:
            # point is not assigned to cluster
            if clusters[i] == -1:
                clusters[i] = last_cluster_marker
                last_cluster_marker += 1
            current_cluster_marker = clusters[i]
            for j in range(n):
                if i == j:
                    continue
                if euclidean_dist(x[i], y[i], x[j], y[j]) < eps:
                    clusters[j] = current_cluster_marker

    # Assign yellow points to clusters
    for i in range(n):
        if flags[i] == Flag.YELLOW.value:
            for j in range(n):
                if i == j:
                    continue
                if euclidean_dist(x[i], y[i], x[j], y[j]) < eps and clusters[j] != -1:
                    clusters[i] = clusters[j]

    plt.figure(2)
    # plot clustered points
    for i in range(n):
        if clusters[i] == -1:
            plt.scatter(x[i], y[i], color='r', marker='*', s=60)
        else:
            cluster_color = clusters[i] / last_cluster_marker
            plt.scatter(x[i], y[i], color=(cluster_color, 0.2, cluster_color ** 2))
            plt.text(x[i], y[i], str(clusters[i]))
    plt.show()
