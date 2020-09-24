import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv('./sunspots.csv').head(100)
    date = list(data.Date)
    emission = list(data['Monthly Mean Total Sunspot Number'])
    emission_new = []
    alpha = 0.5
    emission_new.append(emission[0])
    # Exponential smooth the source range
    for i in range(1, len(date)):
        emission_new.append(alpha*emission[i] + (1 - alpha)*emission_new[i - 1])
    plt.figure(figsize=[30, 10])
    plt.xticks(rotation=90)
    plt.plot(date, emission_new, color='b')
    plt.scatter(date, emission_new, color='b')
    plt.scatter(date, emission, color='r')
    plt.show()
