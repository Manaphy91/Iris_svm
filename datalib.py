import csv
import random
import sys
import os.path

def handle_input():
    if len(sys.argv[1:]) == 0:
        print("Error: no argument has been provided! Please provide the path of the \
            dataset you chose to use.")
        sys.exit(1)
    else:
        if not os.path.exists(sys.argv[1]):
            print("Error: file name provided not related to a file!")
            sys.exit(1)
        else:
            return sys.argv[1]

def read_dataset_from_csv(csv_name, delimiter=","):
    lst = []
    try:
        with open(csv_name, "r") as src:
            reader = csv.reader(src, delimiter=delimiter)
            for line in reader:
                lst.append(line)
    except OSError:
        print("Error: the file named " + csv_name + " can't be opened!")

    return lst

def get_results(dataset):
    lst = []
    for r in dataset:
        lst.append(r[-1])
    return lst

def split_dataset(dataset):
    attr, res = [], []

    for i in range(len(dataset)):
        attr.append(dataset[i][:-1])
        res.append(dataset[i][-1])

    return attr, res

def strings_to_numbers(dataset):
    if len(dataset) ==  0:
        return []
    for column in range(len(dataset[0])):
        for row in range(len(dataset)):
            if len(dataset[row][column]) > 0:
                try:
                    new_val = float(dataset[row][column])
                    dataset[row][column] = new_val
                except ValueError:
                    break

    return dataset

def shuffle_and_split_dataset(dataset):
    ''' Peforms shuffling and splitting of dataset using the Pareto law: 80%
    usable for training and the 20% remaining for testing.
    Returns two arrays containing 80% and 20% of original dataset
    provided as argument.
    '''
    split_ok = False

    SEED = 43
    random.seed(SEED)

    while not split_ok:

        random.shuffle(dataset)

        items = len(dataset)
        training = round(items * 0.80)
        test = items - training

        tr_set = dataset[:training]
        te_set = dataset[training:]

        result_column = set(get_results(dataset))

        if result_column == set(get_results(tr_set)) and \
            result_column == set(get_results(te_set)):
            print("Splitting performed not balanced: some outcomes are missing "
                + "from one of the sets! Perform a new splitting!")
            split_ok = True

    return tr_set, te_set
