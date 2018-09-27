#!/usr/bin/python3 -W ignore

import datalib
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.metrics import make_scorer, accuracy_score, precision_score, \
recall_score, f1_score


if __name__ == "__main__":

    learner = svm.SVC(kernel='linear', C=1)

    training, testing = datalib.shuffle_and_split_dataset(
        datalib.read_dataset_from_csv('iris.data'))

    mat, res = datalib.split_dataset(training)

    mat = datalib.strings_to_numbers(mat)

    learner.fit(mat, res)

    mat, res = datalib.split_dataset(testing)

    predicted = learner.predict(mat)

    print("Dataset name: {}\nKernel type: {}\nC parameter value: {}"
        .format("iris.data", "linear", 1))
    print("Accuracy: {}".format(learner.score(mat, res)))
    print("Classification report:\n")
    print(classification_report(res, predicted))
