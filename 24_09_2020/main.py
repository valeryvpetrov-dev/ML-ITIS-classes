import numpy as np
import matplotlib.pyplot as plt


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


if __name__ == '__main__':
    n, k = 100, 4
    x = np.random.randint(1, 100, n)
    y = np.random.randint(1, 100, n)
    x_cc = np.mean(x)
    y_cc = np.mean(y)
    # find max distance between circle center and all points
    r = []
    for i in range(0, n):
        r.append(dist(x[i], y[i], x_cc, y_cc))
    R = max(r)
    # calculate cluster centers
    x_cluster_center, y_cluster_center = [], []
    for i in range(0, k):
        x_cluster_center.append(R * np.cos(2 * np.pi * i / k) + x_cc)
        y_cluster_center.append(R * np.sin(2 * np.pi * i / k) + y_cc)
    # plot
    plt.scatter(x, y)
    plt.scatter(x_cluster_center, y_cluster_center, color='r')
    plt.show()
