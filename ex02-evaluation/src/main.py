import argparse
import logging
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def load_data_MNTest(fl="./MCTestData.csv"):
    """
    Loads data stored in McNemarTest.csv
    :param fl: filename of csv file
    :return: labels, prediction1, prediction2
    """
    data = pd.read_csv(fl, header=None).to_numpy()
    labels = data[:, 0]
    prediction_1 = data[:, 1]
    prediction_2 = data[:, 2]
    return labels, prediction_1, prediction_2


def load_data_TMStTest(fl="./TMStTestData.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: y1, y2
    """
    data = np.loadtxt(fl, delimiter=",")
    y1 = data[:, 0]
    y2 = data[:, 1]
    return y1, y2


def load_data_FTest(fl="./FTestData.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: evaluations
    """
    errors = np.loadtxt(fl, delimiter=",")
    return errors


def McNemar_test(labels, prediction_1, prediction_2):
    """
    H0: model1 and model2 have the same performance
    H1: performances are not equal
    :param labels: the ground truth labels
    :param prediction_1: the prediction results from model 1
    :param prediction_2:  the prediction results from model 2
    :return: the test statistic chi2_Mc
    """
    m1 = (labels==prediction_1)
    m2 = (labels==prediction_2)
    B = np.sum(np.logical_and(m1, np.logical_not(m2)))
    C = np.sum(np.logical_and(m2, np.logical_not(m1)))
    chi2_Mc = (abs(B-C)-1)**2/(B+C)
    return chi2_Mc


def  TwoMatchedSamplest_test(y1, y2):
    """
    H0: model1 and model2 have the same performance
    H1: performances are not equal
    :param y1: runs of algorithm 1
    :param y2: runs of algorithm 2
    :return: the test statistic t-value
    """
    d = y1-y2
    n_test = len(y1)
    d_mean = np.mean(d)
    d_var  = np.sqrt(1.0/(n_test-1)*np.sum(np.square(d-d_mean)))
    t_value = np.sqrt(n_test)*d_mean/d_var
    return t_value


def Friedman_test(errors):
    """
    H0: all algorithms are equivalent in their performance and hence their average ranks
    H1: the average ranks for at least one algorithm is different
    :param errors: the error values of different algorithms on different datasets
    :return: chi2_F: the test statistic chi2_F value
    :return: FData_stats: the statistical data of the Friedan test data, you can add anything needed to facilitate
    solving the following post hoc problems
    """
    n, k = errors.shape
    rtotal_mean = np.mean(errors)
    r_mean = np.mean(errors, axis=0)
    ss_total = n*np.sum(np.square(r_mean-rtotal_mean))
    ss_error = 1.0/n/(k-1) * np.sum(np.square(errors-rtotal_mean))
    chi2_F = ss_total/ss_error
    FData_stats = {'errors': errors, 'Rj': r_mean}
    return chi2_F, FData_stats


def Nemenyi_test(FData_stats):
    """
    Compares all pairs of algorithm to find best-performing algorithm after H0 of the Friedman-test was rejected
    :param FData_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    :return: the test statisic Q value
    """
    r_mean = FData_stats['Rj']
    errors = FData_stats['errors']
    n, k = errors.shape
    Q_value = np.zeros((k, k), dtype=np.float16)
    inv = 1.0/np.sqrt(k*(k+1)/6/n)
    for i in range(k):
        for j in range(i,k):
            q = (r_mean[i]-r_mean[j])
            Q_value[i,j] = q*inv
    return Q_value


def box_plot(FData_stats):
    """
    TODO
    :param FData_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    """
    pass


def main(args):
    # (a)
    labels, prediction_A, prediction_B = load_data_MNTest()
    chi2_Mc = McNemar_test(labels, prediction_A, prediction_B)

    # (b)
    y1, y2 = load_data_TMStTest()
    t_value = TwoMatchedSamplest_test(y1, y2)

    # (c)
    errors = load_data_FTest()
    chi2_F, FData_stats = Friedman_test(errors)

    # (d)
    Q_value = Nemenyi_test(FData_stats)

    # (e)
    box_plot(FData_stats)


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('ex03')

    cmdline_parser.add_argument('-v', '--verbose', default='INFO', choices=['INFO', 'DEBUG'], help='verbosity')
    cmdline_parser.add_argument('--seed', default=12345, help='Which seed to use', required=False, type=int)
    args, unknowns = cmdline_parser.parse_known_args()
    np.random.seed(args.seed)
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
    main(args)
