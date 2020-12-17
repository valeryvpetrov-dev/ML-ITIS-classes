import pandas as pd
import plotly.graph_objects as go
from numpy import array
from scipy.cluster.vq import kmeans2
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import svm


def read_data():
    data = pd.read_csv('data.csv', sep=',')
    features = array(list(data.values))
    min_max_scaler = MinMaxScaler()
    whitened = min_max_scaler.fit_transform(features)
    init_centroids = [[0, 0, 0], [1, 1, 1]]
    centroid, label = kmeans2(whitened, init_centroids, minit='matrix')
    df = pd.DataFrame(data=whitened, columns=["s1", "s2", 's3'])
    df['label'] = label
    return df


def plot_classification(df, clf):
    figure = go.Figure()
    for C in list(df.label.unique()):
        figure.add_trace(go.Scatter3d(x=df[df.label == C]['s1'],
                                      y=df[df.label == C]['s2'],
                                      z=df[df.label == C]['s3'],
                                      mode='markers',
                                      marker_size=8, marker_line_width=1,
                                      name=f'Cluster {C}'))
    figure.update_layout(width=800, height=800, autosize=True, showlegend=True,
                         scene=dict(xaxis=dict(title='s1', titlefont_color='black'),
                                    yaxis=dict(title='s2', titlefont_color='black'),
                                    zaxis=dict(title='s3', titlefont_color='black')),
                         font=dict(family="Gilroy", color='black', size=12))

    # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
    # Solve for w3 (z)
    z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]
    tmp = np.linspace(0, 1, 30)
    x, y = np.meshgrid(tmp, tmp)
    Z = z(x, y)
    figure.add_surface(x=x, y=y, z=Z, showscale=False)
    figure.show()


if __name__ == '__main__':
    df = read_data()
    X = list(df[['s1', 's2', 's3']].to_records(index=False))
    X = list(map(lambda x: (x[0], x[1], x[2]), X))
    Y = df['label'].tolist()
    model = svm.SVC(kernel='linear')
    clf = model.fit(X, Y)
    tmp1 = X[np.logical_or(Y == 0, Y == 1)]
    tmp2 = Y[np.logical_or(Y == 0, Y == 1)]
    plot_classification(df, clf)
