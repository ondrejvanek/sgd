import csv
import random
import matplotlib.pyplot as plt
from utils import signum, update_weights, compute_total_error

TRAINDATA = "../data/linsep-traindata.csv"
TRANCLASS = "../data/linsep-trainclass.csv"

rand = random.random()


def get_data(filename, append_one=False):
    """
    gets the data from a file name.
    :param filename: the name of the file with data.
    :param append_one: if 1 should be appended at the end of the sample, for offset/intercept consideration.
    :return: dataset - list of lists.
    """
    data = []
    with open(filename, 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            row_int = [float(i) for i in row]
            if append_one:
                row_int.append(1)
            data.append(row_int)
    return data


def sgd(train_data, train_class):
    """
    Runs SGD on training data.
    :param train_data: data with samples - X
    :param train_class: data with classes for each sample - Y.
    :return: returns weights computed by Stochastic Gradient Descent.
    """
    weights = train_data[0]
    rate = 0.001
    changed = False

    errors = 100
    while errors > 0:
        errors = 0
        for i in range(0, len(train_data)):
            index = i  # random.randint(0, len(train_data)-1) - we can either take sequence or random samples.
            sample = train_data[index]
            clazz = int(train_class[index][0])

            result = sum(i[0] * i[1] for i in zip(sample, weights))

            sign = signum(result)

            if sign != clazz:  # there was an error in classification - update it
                changed = True
                error = signum(clazz - sign)
                weights = update_weights(weights, sample, error, rate)
                errors += 1

        print "Errors: ", errors
        if not changed:
            return weights

    return weights


def plot_data():
    m = -weights[0] / weights[1]
    c = weights[2] / weights[1]

    x_1 = [i[0] for i, j in zip(train_data, train_class) if signum(j[0]) == 1]
    x_2 = [i[0] for i, j in zip(train_data, train_class) if signum(j[0]) == -1]
    y_1 = [i[1] for i, j in zip(train_data, train_class) if signum(j[0]) == 1]
    y_2 = [i[1] for i, j in zip(train_data, train_class) if signum(j[0]) == -1]

    plt.plot(x_1, y_1, 'ro')
    plt.plot(x_2, y_2, 'bs')

    x = range(-5, 8, 1)
    y = [m * i + c for i in x]
    plt.plot(x, y, 'k:')

    plt.show()


if __name__ == '__main__':
    train_data = get_data(TRAINDATA, True)
    train_class = get_data(TRANCLASS)

    weights = sgd(train_data, train_class)

    total_error = compute_total_error(train_data, train_class, weights)

    print "Total error: ", total_error, "/", len(train_data)

    plot_data()
