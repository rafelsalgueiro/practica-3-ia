import random
import treepredict
from typing import Union, List


def train_test_split(dataset, test_size: Union[float, int], seed=None):
    if seed:
        random.seed(seed)

    # If test size is a float, use it as a percentage of the total rows
    # Otherwise, use it directly as the number of rows in the test dataset
    n_rows = len(dataset)
    if float(test_size) != int(test_size):
        test_size = int(n_rows * test_size)  # We need an integer number of rows

    # From all the rows index, we get a sample which will be the test dataset
    choices = list(range(n_rows))
    test_rows = random.choices(choices, k=test_size)

    test = [row for (i, row) in enumerate(dataset) if i in test_rows]
    train = [row for (i, row) in enumerate(dataset) if i not in test_rows]

    return train, test


def get_accuracy(classifier, dataset):
    correct = 0
    for row in dataset:
        if treepredict.classify(classifier, row) == row[-1]:
            correct += 1
    return correct / len(dataset)


def mean(values: List[float]):
    return sum(values) / len(values)


def cross_validation(dataset, k, agg, seed, scoref, beta, threshold):
    random.seed(seed)
    random.shuffle(dataset)
    folds = []
    for i in range(k):
        folds.append(dataset[i::k])
    accuracies = []
    for i in range(k):
        train = []
        for j in range(k):
            if j != i:
                train += folds[j]
        test = folds[i]
        tree = treepredict.buildtree(train, scoref, beta)
        treepredict.prune(tree, threshold)
        accuracy = get_accuracy(tree, test)
        accuracies.append(accuracy)
    return mean(accuracies)
