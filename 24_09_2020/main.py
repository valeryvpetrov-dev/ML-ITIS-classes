import numpy as np

if __name__ == '__main__':
    n, k = 100, 4
    x = np.random.randint(1, 100, n)
    y = np.random.randint(1, 100, n)
    x_cc = np.mean(x)
    y_cc = np.mean(y)
    print(x)
    print(x_cc)
