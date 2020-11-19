import pandas as pd
import plotly.graph_objects as go
from numpy import array
from scipy.cluster.vq import kmeans2
from sklearn.preprocessing import MinMaxScaler


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


def plot_data(df):
    plot = go.Figure()
    for C in list(df.label.unique()):
        plot.add_trace(go.Scatter3d(x=df[df.label == C]['s1'],
                                    y=df[df.label == C]['s2'],
                                    z=df[df.label == C]['s3'],
                                    mode='markers',
                                    marker_size=8, marker_line_width=1,
                                    name=f'Cluster {C}'))

    plot.update_layout(width=800, height=800, autosize=True, showlegend=True,
                       scene=dict(xaxis=dict(title='s1', titlefont_color='black'),
                                  yaxis=dict(title='s2', titlefont_color='black'),
                                  zaxis=dict(title='s3', titlefont_color='black')),
                       font=dict(family="Gilroy", color='black', size=12))
    plot.show()


if __name__ == '__main__':
    df = read_data()
    plot_data(df)
    # максимизировать log...
    # градиентыный спуск - wikibooks

