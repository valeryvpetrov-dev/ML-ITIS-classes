import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_data(file):
    return pd.read_csv(file)


def label_rects(rects, ax):
    total_height = sum(rect.get_height() for rect in rects)
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            str("{} ({})".format(height, round(height / total_height, 2))),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 0),
            textcoords='offset points',
            ha='center', va='bottom'
        )


def plot_died_vs_survived(data):
    # prepare data to plot
    survived = data.Survived.value_counts()
    survived = survived.rename(index={0: 'died', 1: 'survived'})
    # configure plot
    ticks_x = survived.index
    figure, axes = plt.subplots()
    rects_survived = axes.bar(ticks_x, survived)
    label_rects(rects_survived, axes)
    axes.set_title("Died vs survived")
    # save plot to file
    plt.savefig('./plot/died_vs_survived.png')
    # show plot
    plt.show()


if __name__ == '__main__':
    data = read_data(open('titanic.csv'))
    # how many people died/survived
    plot_died_vs_survived(data)
