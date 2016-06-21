
def signum(number):
    """
    Substitution for the math.sgn
    :param number: number for which the sgn function should be computed.
    :return: -1, 0, 1, depending on the signum function.
    """

    if number > 0:
        return 1
    elif number < 0:
        return -1

    return 0


def update_weights(weights, sample, error, rate):
    """
    Update weights mechanism.
    :param weights: the weights to be updated.
    :param sample: sample which should be used to update weights.
    :param error: error of the sample.
    :param rate: rate of change, or coefficient of learning.
    :return: updated weights in the same format as weights.
    """

    weighted_sample = [i * rate * error for i in sample]

    return [w + s for s, w in zip(weighted_sample, weights)]


def compute_total_error(X_data, Y_data, weights):
    """
    Computes total error on the data - given the loss function is 1 per misclassified sample independent on class.
    :param X_data: samples.
    :param Y_data: classes.
    :return: total error on the data.
    """
    total_error = 0
    for i in range(0, len(X_data)):
        index = i
        sample = X_data[index]
        clazz = int(Y_data[index][0])

        result = sum(i[0] * i[1] for i in zip(sample, weights))

        sign = signum(result)

        if sign != clazz:
            total_error += 1

    return total_error

