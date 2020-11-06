import pandas as pd


def read_data():
    return pd.read_csv('./data/symptom_disease.csv')


if __name__ == '__main__':
    data_df = read_data()
    print(data_df)
