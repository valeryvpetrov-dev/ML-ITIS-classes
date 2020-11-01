import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

viridis = cm.get_cmap('viridis', 3)


def map_species(name):
    if name == 'Iris-setosa':
        return 0
    elif name == 'Iris-versicolor':
        return 1
    elif name == 'Iris-virginica':
        return 2
    else:
        return -1


def read_iris_data(file):
    df = pd.read_csv(file)
    df = df[['SepalLengthCm', 'SepalWidthCm', 'Species']]
    df['Species'] = df['Species'].map(map_species)
    return df


def plot_knn(df_train, df_test_control, df_test_actual, K, accuracy):
    df_train = df_train.sort_values(by=['Species'])
    df_test_control = df_test_control.sort_values(by=['Species'])
    df_test_actual = df_test_actual.sort_values(by=['Species'])
    ax1 = df_train.plot.scatter(x='SepalLengthCm', y='SepalWidthCm', c='Species', colormap='viridis', marker='o')
    ax2 = df_test_control.plot.scatter(x='SepalLengthCm', y='SepalWidthCm', c='Species', colormap='viridis', marker=6,
                                       s=70, colorbar=False, ax=ax1)
    ax3 = df_test_actual.plot.scatter(x='SepalLengthCm', y='SepalWidthCm', c='Species', colormap='viridis', marker=7,
                                      s=70, colorbar=False, ax=ax1)
    plt.title("K={}, Accuracy={}".format(K, accuracy))
    plt.show()


def split_train_test(df, train_percent=0.8):
    mask = np.random.rand(len(df)) < train_percent
    return df[mask], df[~mask]


def euclidean_dist(iris_1, iris_2):
    return np.sqrt((iris_1[0] - iris_2[0]) ** 2 + (iris_1[1] - iris_2[1]) ** 2)


def classify_knn(df_train, df_test, K):
    df_test_actual = df_test.copy()
    for i, test in df_test.iterrows():
        df_test_to_train = pd.DataFrame(columns=['Dist', 'Species'], index=df_train.index)
        for j, train in df_train.iterrows():
            current_dist = euclidean_dist(test, train)
            df_test_to_train.at[j, 'Dist'] = current_dist
            df_test_to_train.at[j, 'Species'] = train['Species']
        knn = df_test_to_train.sort_values(by=['Dist']).head(K)
        knn_count = knn.groupby(['Species']).count()
        knn_max_count = int(knn_count.idxmax(axis=0)[0])
        df_test_actual.at[i, 'Species'] = knn_max_count
    return df_test_actual


def calculate_accuracy(df_test_control, df_test_actual):
    right_count = 0
    for i, test_control in df_test_control.iterrows():
        if test_control['Species'] == df_test_actual.at[i, 'Species']:
            right_count += 1
    return right_count / len(df_test_control)


if __name__ == '__main__':
    df = read_iris_data('./data/Iris.csv')
    df_train, df_test = split_train_test(df)
    K = 20  # number of nearest neighbours to consider
    df_test_actual = classify_knn(df_train, df_test, K)
    accuracy = calculate_accuracy(df_test, df_test_actual)
    plot_knn(df_train, df_test, df_test_actual, K, accuracy)
