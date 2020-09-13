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


def plot_died_vs_survived_wrt_sex(data):
    # prepare data to plot
    survived_men = data.loc[data.Sex == 'male'].Survived.value_counts().sort_index()
    survived_women = data.loc[data.Sex == 'female'].Survived.value_counts().sort_index()
    # configure plot
    survived_ticks = survived_men.index.values
    ticks_x = survived_ticks
    ticks_x_labels = map(lambda tick: 'died' if tick == 0 else 'survived', survived_ticks)
    figure, axes = plt.subplots()
    bar_width = 0.4
    rects_survived_men = axes.bar(ticks_x - bar_width / 2, survived_men, bar_width, label='Men')
    rects_survived_women = axes.bar(ticks_x + bar_width / 2, survived_women, bar_width, label='Women')
    label_rects(rects_survived_men, axes)
    label_rects(rects_survived_women, axes)
    axes.set_title("Died vs survived wrt sex")
    axes.set_xticks(ticks_x)
    axes.set_xticklabels(ticks_x_labels)
    axes.legend()
    # save plot to file
    plt.savefig('./plot/died_vs_survived_wrt_sex.png')
    # show plot
    plt.show()


def plot_family_aboard_survival(data):
    # prepare data
    sibsp_survival = data[['Survived', 'SibSp']].groupby('SibSp').mean()
    parch_survival = data[['Survived', 'Parch']].groupby('Parch').mean()

    family_data = data[['Survived', 'SibSp', 'Parch']]
    family_data.insert(0, 'Family', data['SibSp'] + data['Parch'])
    family_survival = family_data \
        .drop('SibSp', 1).drop('Parch', 1) \
        .groupby('Family').mean()

    # configure plot
    data_to_plot = [
        (sibsp_survival, "Number of siblings and spouses aboard"),
        (parch_survival, "Number of parents and children aboard"),
        (family_survival, "Number of family members aboard"),
    ]
    number_of_plots = len(data_to_plot)
    ticks_y = np.linspace(0, 1, 5)
    for i in range(number_of_plots):
        axes = plt.subplot2grid((number_of_plots, 1), (i, 0))
        x = data_to_plot[i][0].index
        y = data_to_plot[i][0]
        axes.plot(x, y)
        # configure plot
        axes.set_xticks(x.values)
        axes.set_yticks(ticks_y)
        axes.set_xlabel(data_to_plot[i][1])
        axes.set_ylabel("Survival")
    # save pot to file
    plt.tight_layout()
    plt.savefig('./plot/family_aboard_survival.jpg')
    # show plot
    plt.show()


if __name__ == '__main__':
    data = read_data(open('titanic.csv'))
    # how many people died/survived
    plot_died_vs_survived(data)
    # how many people died/survived wrt sex
    plot_died_vs_survived_wrt_sex(data)
    # relation between number family members to survival
    plot_family_aboard_survival(data)
