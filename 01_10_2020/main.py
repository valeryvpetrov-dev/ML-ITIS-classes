import numpy as np
import matplotlib.pyplot as plt
from random import uniform

EPS = 0.1
points_number = 100
clusters_number = 3
m = 1.5  # uncertainty factor


def euclidean_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def assign_points_to_clusters(U: list) -> list:
    points_to_clusters = []
    for i in range(points_number):
        max_i_U = 0
        cluster = 0
        for j in range(clusters_number):
            current_i_U = U[j][i]
            if current_i_U > max_i_U:
                max_i_U = current_i_U
                cluster = j
        points_to_clusters.append(cluster)
    return points_to_clusters


def calculate_sum_cluster_distance(clusters_number, x_points, y_points, x_cluster_centers, y_cluster_centers,
                                   points_to_clusters):
    sum_distance_to_clusters_centers = 0
    for i in range(clusters_number):
        for j in range(points_number):
            if points_to_clusters[j] == i:  # for each point in specific cluster
                distance_to_cluster_center = euclidean_dist(x_points[j], y_points[j], x_cluster_centers[i],
                                                            y_cluster_centers[i])
                sum_distance_to_clusters_centers += distance_to_cluster_center
    return sum_distance_to_clusters_centers


def calculate_clusters_centers(x_points: list, y_points: list, U: list, m: int) -> tuple:
    x_clusters_centers = []
    y_clusters_centers = []
    for i in range(clusters_number):
        U_c = U[i]
        x_numerator = 0
        y_numerator = 0
        denominator = 0
        for j in range(points_number):
            point_U_c = pow(U_c[j], m)
            x_numerator += point_U_c * x_points[j]
            y_numerator += point_U_c * y_points[j]
            denominator += point_U_c
        x_clusters_centers.append(x_numerator / denominator)
        y_clusters_centers.append(y_numerator / denominator)
    return x_clusters_centers, y_clusters_centers


def calculate_clusters_dist(x_points: list, y_points: list, x_cluster_centers: list, y_cluster_centers: list) -> list:
    return [
        [
            euclidean_dist(
                x_points[j], y_points[j], x_cluster_centers[i], y_cluster_centers[i]
            ) for j in range(points_number)
        ] for i in range(clusters_number)
    ]


def calculate_affiliation_matrix(dist_clusters_centers: list, m: int) -> list:
    power = 2 / (m - 1)
    # calculate
    return [
        [
            pow(1 / current_dist, power) for current_dist in dist_clusters_centers[i]
        ] for i in range(clusters_number)
    ]


def normalize_affiliation_matrix(U: list):
    # normalize
    for i in range(points_number):
        current_sum = 0
        for j in range(clusters_number):
            current_sum += U[j][i]
        for j in range(clusters_number):
            U[j][i] /= current_sum


def calculate_decision_function(dist_clusters_centers: list, U: list) -> float:
    decision_value = 0
    for i in range(clusters_number):
        for j in range(points_number):
            decision_value += dist_clusters_centers[i][j] * U[i][j]
    return decision_value


def plot_clustering(
        clustering_step: int,
        x_points: list, y_points: list, points_to_clusters: list,
        x_clusters_centers: list, y_clusters_centers: list
):
    # plot cluster centers
    plt.figure(clustering_step)
    clusters_colors = plt.cm.get_cmap('hsv', clusters_number + 1)
    for i in range(0, clusters_number):
        plt.scatter(x_cluster_centers[i], y_cluster_centers[i], color=clusters_colors(i), marker='*', s=200)

    # plot current c_means_steps
    for i in range(0, points_number):
        current_point_x = x_points[i]
        current_point_y = y_points[i]
        current_point_cluster = points_to_clusters[i]
        plt.scatter(current_point_x, current_point_y, color=clusters_colors(current_point_cluster))


if __name__ == '__main__':
    # generate input points
    x_points = np.random.randint(1, 100, points_number).tolist()
    y_points = np.random.randint(1, 100, points_number).tolist()
    # plot input points
    # plt.scatter(x_points, y_points)

    # affiliation matrix
    U = [[uniform(0, 1) for x in range(points_number)] for y in range(clusters_number)]
    normalize_affiliation_matrix(U)
    # cluster until new centers are not stand still
    clustering_step = 1
    prev_decision_value, current_decision_value = None, None
    clustering_done = False
    while not clustering_done:
        print('Clustering step: {}'.format(clustering_step))
        # calculate cluster centers
        x_cluster_centers, y_cluster_centers = calculate_clusters_centers(x_points, y_points, U, m)
        dist_clusters_centers = calculate_clusters_dist(x_points, y_points, x_cluster_centers, y_cluster_centers)
        U = calculate_affiliation_matrix(dist_clusters_centers, m)
        normalize_affiliation_matrix(U)
        # make decision
        current_decision_value = calculate_decision_function(dist_clusters_centers, U)
        if prev_decision_value is None:
            prev_decision_value = current_decision_value
        else:
            diff_decision_values = abs(prev_decision_value - current_decision_value)
            if diff_decision_values < EPS:
                clustering_done = True
            print('Decisions difference: {}'.format(diff_decision_values))
            prev_decision_value = current_decision_value
        clustering_step += 1

    points_to_clusters = assign_points_to_clusters(U)
    plot_clustering(clustering_step, x_points, y_points, points_to_clusters, x_cluster_centers, y_cluster_centers)
    print('C-means clustering done in {} steps'.format(clustering_step))
