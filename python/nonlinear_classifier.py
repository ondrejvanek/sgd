import csv
import random
import matplotlib.pyplot as plt
import math

from utils import signum, update_weights, compute_total_error

TRAINDATA = "../data/nonlinsep-traindata.csv"
TRANCLASS = "../data/nonlinsep-trainclass.csv"

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
            row_float = [float(i) for i in row]
            if append_one:
                row_float.append(1)
                row_float.append(row_float[0] ** 2)
                row_float.append(row_float[1] ** 2)
                row_float.append(row_float[1] * row_float[0])

            data.append(row_float)
    return data


def sgd(train_data, train_class):
    """
    Runs SGD on training data.
    :param train_data: data with samples - X
    :param train_class: data with classes for each sample - Y.
    :return: returns weights computed by Stochastic Gradient Descent.
    """
    weights = train_data[0]
    rate = 0.01
    changed = True

    wrong = train_data
    wrong_class = [int(i[0]) for i in train_class]

    while changed:
        changed = False
        for i in range(0, len(wrong)):
            index = i  # random.randint(0, len(train_data)-1) - we can either take sequence or random samples.
            sample = wrong[index]
            clazz = int(wrong_class[index])

            result = sum(i[0] * i[1] for i in zip(sample, weights))

            sign = signum(result)

            if sign != clazz:  # there was an error in classification - update it
                changed = True
                error = signum(clazz - sign)
                weights = update_weights(weights, sample, error, rate)

        if len(wrong) < 1:
            return weights

        wrong, wrong_class = get_wrong(train_data, train_class, weights)

        print "Wrong samples: ", len(wrong)

    return weights


def get_wrong(X_data, Y_data, weights):
    """
    gets all wrong samples and their classes
    :param X_data: samples
    :param Y_data: classes
    :param weights: current weights
    :return:
    """
    wrong = []
    wrong_class = []
    for i in range(0, len(X_data)):
        sample = X_data[i]
        clazz = int(Y_data[i][0])

        result = sum(i[0] * i[1] for i in zip(sample, weights))

        sign = signum(result)

        if sign != clazz:
            wrong.append(sample)
            wrong_class.append(clazz)

    return wrong, wrong_class


def quadratic(weights, x):
    """
    Computed the values for given sample and weights
    :param weights: the weights for the polyline equation
    :param x: the point on the x axis
    :return: value on the y axis given x and parameters of the polyline
    """

    try:
        a = weights[4]
        b = weights[5] * x + weights[1]
        c = weights[0] * x + weights[2] + weights[3] * x * x

        discRoot = math.sqrt((b * b) - 4 * a * c)
        root1 = (-b + discRoot) / (2 * a)
        root2 = (-b - discRoot) / (2 * a)

        return [root1, root2]
    except ValueError:
        return 0, 0


def plot_data(weights, train_data, train_class):
    """
    Plots data for the polynomial SGD.
    :param weights: weights of the perceptron
    :param train_data: samples
    :param train_class: classes for samples
    :return: nothing.
    """

    x_1 = [i[0] for i, j in zip(train_data, train_class) if signum(j[0]) == 1]
    x_2 = [i[0] for i, j in zip(train_data, train_class) if signum(j[0]) == -1]
    y_1 = [i[1] for i, j in zip(train_data, train_class) if signum(j[0]) == 1]
    y_2 = [i[1] for i, j in zip(train_data, train_class) if signum(j[0]) == -1]

    plt.plot(x_1, y_1, 'ro')
    plt.plot(x_2, y_2, 'gs')

    X = range(-400, 400)

    X = [x / 100.0 for x in X]

    Y_1 = [quadratic(weights, x)[0] for x in X]
    Y_2 = [quadratic(weights, x)[1] for x in X]

    plt.plot(X, Y_1, 'k:')
    plt.plot(X, Y_2, 'b:')

    plt.show()


if __name__ == '__main__':
    train_data = get_data(TRAINDATA, True)
    train_class = get_data(TRANCLASS)

    weights = sgd(train_data, train_class)

    total_error = compute_total_error(train_data, train_class, weights)

    plot_data(weights, train_data, train_class)

    print "Total error: ", total_error, "/", len(train_data)
