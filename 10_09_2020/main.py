import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_csv(file):
    data = pd.read_csv(file, usecols=['gender', 'age'])
    return data


if __name__ == '__main__':
    data = read_csv(open('data.csv'))
    print(data['gender'])
    print(data['age'])
