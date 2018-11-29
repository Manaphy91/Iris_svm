#!/usr/bin/python3 -W ignore

import datalib
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.metrics import make_scorer, accuracy_score, precision_score, \
recall_score, f1_score


if __name__ == "__main__":

    training, testing = datalib.shuffle_and_split_dataset(
        datalib.read_dataset_from_csv('winequality-red.csv'))

    mat, res = datalib.split_dataset(training)

    learner = svm.SVC(kernel='rbf', gamma=10, C=1)

    mat = datalib.strings_to_numbers(mat)

    learner.fit(mat, res)

    mat, res = datalib.split_dataset(testing)

    predicted = learner.predict(mat)

    print("Dataset name: {}\nKernel type: {}\ngamma parameter value: {}\
        \nC parameter value: {}".format("winequality-red.csv", "rbf", 10, 1))
    print("Accuracy: {}".format(learner.score(mat, res)))
    print("Classification report:\n")
    print(classification_report(res, predicted))
