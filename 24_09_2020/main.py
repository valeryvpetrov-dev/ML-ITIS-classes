import numpy as np
import matplotlib.pyplot as plt


def euclidean_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def assign_points_to_clusters(x_points, y_points, x_cluster_centers, y_cluster_centers):
    points_number = len(x_points)
    clusters_number = len(x_cluster_centers)
    points_to_cluster = []
    for i in range(0, points_number):
        min_dist_to_cluster = None
        min_dist_cluster = None
        for j in range(0, clusters_number):
            curr_dist_to_cluster = euclidean_dist(x_points[i], y_points[i], x_cluster_centers[j], y_cluster_centers[j])
            if min_dist_to_cluster is None or curr_dist_to_cluster < min_dist_to_cluster:
                min_dist_to_cluster = curr_dist_to_cluster
                min_dist_cluster = j
        points_to_cluster.append(min_dist_cluster)
    return points_to_cluster


def recalculate_cluster_centers(x_points, y_points, clusters_number, points_to_clusters):
    # TODO continue
    for i in range(0, clusters_number):
        pass


if __name__ == '__main__':
    points_number, clusters_number = 100, 4
    # generate input points
    x_points = np.random.randint(1, 100, points_number)
    y_points = np.random.randint(1, 100, points_number)
    # plot input points
    plt.scatter(x_points, y_points)

    # assume that cluster center is in the center
    x_cluster_center = np.mean(x_points)
    y_cluster_center = np.mean(y_points)

    # find max distance between cluster center and all points
    distances_to_cluster_center = []
    for i in range(0, points_number):
        distance_to_cluster_center = euclidean_dist(x_points[i], y_points[i], x_cluster_center, y_cluster_center)
        distances_to_cluster_center.append(distance_to_cluster_center)
    max_distance_to_cluster_center = max(distances_to_cluster_center)

    # calculate cluster centers
    x_cluster_centers, y_cluster_centers = [], []
    for i in range(0, clusters_number):
        x_cluster_center = max_distance_to_cluster_center * np.cos(2 * np.pi * i / clusters_number) + x_cluster_center
        y_cluster_center = max_distance_to_cluster_center * np.sin(2 * np.pi * i / clusters_number) + y_cluster_center
        x_cluster_centers.append(x_cluster_center)
        y_cluster_centers.append(y_cluster_center)

    # plot cluster centers
    clusters_colors = plt.cm.get_cmap('hsv', clusters_number + 1)
    for i in range(0, clusters_number):
        plt.scatter(x_cluster_centers[i], y_cluster_centers[i], color=clusters_colors(i), marker='*', s=200)

    # assign each point to cluster
    points_to_clusters = assign_points_to_clusters(x_points, y_points, x_cluster_centers, y_cluster_centers)
    # plot current clusterization
    for i in range(0, points_number):
        current_point_x = x_points[i]
        current_point_y = y_points[i]
        current_point_cluster = points_to_clusters[i]
        plt.scatter(current_point_x, current_point_y, color=clusters_colors(current_point_cluster))
    plt.show()

    # recalculate clusters centers
    x_cluster_centers, y_cluster_centers = recalculate_cluster_centers(x_points, y_points, clusters_number, points_to_clusters)
