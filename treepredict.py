#!/usr/bin/env python3
import sys
import collections
from math import log2
from typing import List, Tuple

# Used for typing
Data = List[List]


def read(file_name: str, separator: str = ",") -> Tuple[List[str], Data]:
    """
    t3: Load the data into a bidimensional list.
    Return the headers as a list, and the data
    """
    header = None
    data = []
    with open(file_name, "r") as fh:
        for line in fh:
            values = line.strip().split(separator)
            if header is None:
                header = values
                continue
            data.append([_parse_value(v) for v in values])
        return header,data

def _parse_value(v):
    try:
        if float(v) == int(v):
            return int(v)
        else:
            return float(v)
    except ValueError:
        return v

def unique_counts(part: Data):
    """
    t4: Create counts of possible results
    (the last column of each row is the
    result) in a dataset                                                                        Las 3 son correctas
    """
    #result = collections.Counter()
    #for row in part:
    #    c = row [-1]
    #    result[c] += 1
    #return dict(result)

    return dict(collections.Counter(row[-1] for row in part))

    #results = {}
    #for row in part:
    #    c = row[-1]
    #    if c not in results:
    #        results[c] = 0
    #    results[c] += 1
    #return results


def gini_impurity(part: Data):
    """
    t5: Computes the Gini index of a node
    """
    total = len(part)
    if total == 0:
        return 0
    results = unique_counts(part)
    imp = 1
    for v in results.values():
        imp -= (v/total)**2
    return imp


def entropy(part: Data):
    """
    t6: Entropy is the sum of p(x)log(p(x))
    across all the different possible results
    """
    total = len(part)
    results = unique_counts(part)
    imp = 0
    return -sum(
        (v/total) * log2(v/total) for v in results.values()
    )


def _split_numeric(prototype: List, column: int, value):
    return prototype[column] >= value


def _split_categorical(prototype: List, column: int, value):
    return prototype[column] == value


def divideset(part: Data, column: int, value: int, set1=None, set2=None) -> Tuple[Data, Data]:
    """
    t7: Divide a set on a specific column. Can handle
    numeric or categorical values
    """
    if isinstance(value, (int, float)):
        split_function = _split_numeric
    else:
        split_function = _split_categorical
    
    for row in part:
        if split_function(row, column, value):
            set1.append(row)
        else:
            set2.append(row)

    return (set1, set2)


class DecisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        """
        t8: We have 5 member variables:
        - col is the column index which represents the
          attribute we use to split the node
        - value corresponds to the answer that satisfies
          the question
        - tb and fb are internal nodes representing the
          positive and negative answers, respectively
        - results is a dictionary that stores the result
          for this branch. Is None except for the leaves
        """
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def buildtree(part: Data, scoref=entropy, beta=0):
    """
    t9: Define a new function buildtree. This is a recursive function
    that builds a decision tree using any of the impurity measures we
    have seen. The stop criterion is max_s\Delta i(s,t) < \beta
    """
    if len(part) == 0:
        return DecisionNode()

    current_score = scoref(part)
    
    if current_score == 0:
        # No further partitioning
        print(unique_counts(part))
        return DecisionNode(results=unique_counts(part))

    best_gain, best_criteria, best_sets = best_params_buildtree(part, scoref)

    if best_gain > beta:
        true_branch = buildtree(best_sets[0], scoref, beta)
        false_branch = buildtree(best_sets[1], scoref, beta)
        return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=true_branch, fb=false_branch)
    else:
        return DecisionNode(results=unique_counts(part))

def best_params_buildtree(part: Data, scoref=entropy):
    current_score = scoref(part)
    best_gain = 0
    best_criteria = None
    best_sets = None

    column_count = len(part[0]) - 1

    for col in range(0, column_count):
        # Generate the list of different values in this column
        column_values = {}
        for row in part:
            column_values[row[col]] = 1

        for value in column_values.keys():
            (set1, set2) = divideset(part, col, value, [], [])      

            # Information gain
            p = len(set1) / len(part)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)

            if gain > best_gain:    #this is the stop criterion
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    
    return best_gain, best_criteria, best_sets


def iterative_buildtree(part: Data, scoref=entropy, beta=0):
    """
    t10: Define the iterative version of the function buildtree
    """
    if len(part) == 0:
        return DecisionNode()

    current_score = scoref(part)

    if current_score == 0:
        # No further partitioning
        return DecisionNode(results=unique_counts(part))
    
    # Set up some variables to track the best criteria
    best_gain = 0
    best_criteria = None
    best_sets = None
    
    best_gain, best_criteria, best_sets = best_params_buildtree(part, scoref)

    return iterative_buildtree(best_sets[0], scoref, beta), iterative_buildtree(best_sets[1], scoref, beta), best_criteria

    

def classify(tree, row):
    raise NotImplementedError


def print_tree(tree, headers=None, indent=""):
    """
    t11: Include the following function
    """
    # Is this a leaf node?
    if tree.results is not None:
        print(tree.results)
    else:
        # Print the criteria
        criteria = tree.col
        if headers:
            criteria = headers[criteria]
        print(f"{indent}{criteria}: {tree.value}?")

        # Print the branches
        print(f"{indent}T->")
        print_tree(tree.tb, headers, indent + "  ")
        print(f"{indent}F->")
        print_tree(tree.fb, headers, indent + "  ")


def print_data(headers, data):
    colsize = 15
    print('-' * ((colsize + 1) * len(headers) + 1))
    print("|", end="")
    for header in headers:
        print(header.center(colsize), end="|")
    print("")
    print('-' * ((colsize + 1) * len(headers) + 1))
    for row in data:
        print("|", end="")
        for value in row:
            if isinstance(value, (int, float)):
                print(str(value).rjust(colsize), end="|")
            else:
                print(value.ljust(colsize), end="|")
        print("")
    print('-' * ((colsize + 1) * len(headers) + 1))

def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "decision_tree_example.txt"
    header, data = read(filename)
    #print_data(header, data)

    print("----------Unique counts----------")
    print(unique_counts(data))
    print(unique_counts([]))
    print(unique_counts([data[0]]))

    print("----------Gini impurity----------")
    print(gini_impurity(data))
    print(gini_impurity([]))
    print(gini_impurity([data[0]]))

    print("----------Entropy----------")
    print(entropy(data))
    print(entropy([]))
    print(entropy([data[0]]))

    print("----------Build tree----------")
    headers, data = read(filename)
    tree = buildtree(data)
    #print_tree(tree, headers)


if __name__ == "__main__":
    main()