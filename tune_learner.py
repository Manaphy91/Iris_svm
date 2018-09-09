#!/usr/bin/python3 -W ignore

import sample
import datalib
from functools import partial
from sklearn.svm import SVC

def print_recap(params_names, params_values, metrics):
    comb = dict(zip(params_names, params_values))
    print("\tBest params: {}\n\tBest metrics: {}".format(
        comb, metrics))

def best_rbf(mat, res):

    C = tuple(sample.generate_pow_range(10, -2, 10))

    gamma = list(sample.generate_pow_range(10, -9, 3))

    # add default value for gamme equal to 1/(number of features)
    gamma.append(round(1/len(mat[0]), 3))

    rbf_svc = partial(SVC, kernel='rbf')

    print("Kernel type: {}".format('rbf'))

    comb, metrics = sample.get_best_params(rbf_svc, mat, res, ('C', 'gamma'), \
        [C, gamma])

    print_recap(('C', 'gamma'), comb, metrics)

def best_linear(mat, res):
    
    C = tuple(sample.generate_pow_range(10, -2, 10))

    linear_svc = partial(SVC, kernel='linear')

    print("Kernel type: {}".format('linear'))

    comb, metrics = sample.get_best_params(linear_svc, mat, res, ('C'), \
        [C])

    print_recap(('C'), comb, metrics)


def best_poly(mat, res):
    
    C = tuple(sample.generate_pow_range(10, -2, 10))

    degree = tuple(range(1, 10))

    poly_svc = partial(SVC, kernel='poly')

    print("Kernel type: {}".format('poly'))

    comb, metrics = sample.get_best_params(poly_svc, mat, res, ('C', 'degree'), \
        [C, degree])

    print_recap(('C', 'degree'), comb, metrics)


if __name__ == "__main__":
    training, testing = datalib.shuffle_and_split_dataset( \
        datalib.read_dataset_from_csv('iris.data'))
    
    mat, res = datalib.split_dataset(training)
    
    mat = datalib.strings_to_numbers(mat)

    best_rbf(mat, res) 

    best_linear(mat, res)

    best_poly(mat, res)
