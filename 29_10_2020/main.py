import operator
import pprint as pp

import numpy as np
import pandas as pd


def read_data():
    return pd.read_csv('./data/symptom_disease.csv')


# classical probability
def calculate_diseases_prob(diseases_number, N):
    return dict(map(lambda kv: (kv[0], kv[1] / N), diseases_number.items()))


# full probability
def calculate_symptoms_prob(symptoms_diseases_df, diseases_prob):
    symptoms_prob = dict()
    for i, row in symptoms_diseases_df.iterrows():
        current_symptom = row['Симптом']
        current_symptom_prob = 0
        for key in diseases_prob.keys():
            current_symptom_prob += diseases_prob[key] * row[key]
        symptoms_prob[current_symptom] = current_symptom_prob
    return symptoms_prob


def calculate_symptoms_diseases_prob(symptoms, symptoms_diseases_df, diseases_prob, symptoms_prob):
    symptoms_diseases_prob = dict()

    # calculate denominator
    denominator = 1
    for symptom_kv in symptoms.items():
        if symptom_kv[1] == 1:
            denominator *= symptoms_prob[symptom_kv[0]]

    # calculate numerator
    for disease_kv in diseases_prob.items():
        current_disease = disease_kv[0]
        numerator = 1
        for symptom_kv in symptoms.items():
            if symptom_kv[1] == 1:
                numerator *= symptoms_diseases_df.loc[
                    symptoms_diseases_df['Симптом'] == symptom_kv[0],
                    current_disease
                ].values[0]
        numerator *= diseases_prob[current_disease]
        symptoms_diseases_prob[current_disease] = numerator / denominator
    return sorted(symptoms_diseases_prob.items(), key=operator.itemgetter(1), reverse=True)


if __name__ == '__main__':
    symptoms_diseases_df = read_data()
    N = 303  # number of diseases analysed
    diseases_number = {
        'Острый правостор паратонз абсцесс': 96,
        'Острый левостор паратонз абсцесс': 116,
        'Острый правостор паратон-зиллит': 26,
        'Острый левостор паратон-зиллит': 30,
        'Острый двухстор паратонз абсцесс': 9,
        'Острый правостор парафарин абсцесс': 3,
        'Острый левостор парафарин абсцесс': 3,
        'Острый правостор парафарингит': 8,
        'Острый левостор парафарингит': 12
    }
    diseases_prob = calculate_diseases_prob(diseases_number, N)
    symptoms_prob = calculate_symptoms_prob(symptoms_diseases_df, diseases_prob)

    symptoms_number = symptoms_diseases_df.shape[1] - 1
    symptoms_presence = np.random.randint(2, size=symptoms_number)
    symptoms = dict(zip(symptoms_diseases_df['Симптом'].values, symptoms_presence))
    print("Current symptoms")
    pp.pprint(symptoms)

    symptoms_diseases_prob = calculate_symptoms_diseases_prob(
        symptoms, symptoms_diseases_df, diseases_prob, symptoms_prob
    )
    print("Diseases probabilities:")
    pp.pprint(symptoms_diseases_prob)
