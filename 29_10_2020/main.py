import pandas as pd
import numpy as np
import pprint as pp


def read_data():
    return pd.read_csv('./data/symptom_disease.csv')


if __name__ == '__main__':
    data_df = read_data()
    symptoms_number = data_df.shape[1] - 1
    symptoms_presence = np.random.randint(2, size=symptoms_number)
    symptoms = dict(zip(data_df.columns[1:], symptoms_presence))
    pp.pprint(symptoms)
