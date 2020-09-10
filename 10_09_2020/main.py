import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_data(file):
    return pd.read_csv(file, usecols=['Sex', 'Age'])


if __name__ == '__main__':
    data = read_data(open('titanic.csv'))
    print(data['Sex'])
    print(data['Age'])
