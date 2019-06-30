import collections
import math

import numpy
import pandas
DEBUG = False


def __extractNodes(matrix):
    nodes = set()
    for colKey in matrix:
        nodes.add(colKey)
    for rowKey in matrix.T:
        nodes.add(rowKey)
    return nodes


def __makeSquare(matrix, keys, default=0.0):
    matrix = matrix.copy()

    def insertMissingColumns(matrix):
        for key in keys:
            if not key in matrix:
                matrix[key] = pandas.Series(default, index=matrix.index)
        return matrix

    matrix = insertMissingColumns(matrix)  # insert missing columns
    matrix = insertMissingColumns(matrix.T).T  # insert missing rows

    return matrix.fillna(default)


def __ensureRowsPositive(matrix):
    matrix = matrix.T
    for colKey in matrix:
        if matrix[colKey].sum() == 0.0:
            matrix[colKey] = pandas.Series(numpy.ones(len(matrix[colKey])), index=matrix.index)
    return matrix.T


def __normalizeRows(matrix):
    return matrix.div(matrix.sum(axis=1), axis=0)


def __euclideanNorm(series):
    return math.sqrt(series.dot(series))


# PageRank specific functionality:

def __startState(nodes, papers_h_index):
    if len(nodes) == 0: raise ValueError("There must be at least one node.")
    sum_h_index = sum(papers_h_index.values())
    return pandas.Series({node: papers_h_index[node]/sum_h_index for node in nodes})

def __integrateRandomSurfer(nodes, transitionProbabilities, rsp, papers_h_index):
    alpha = papers_h_index / numpy.sum(papers_h_index) * rsp
    m1= transitionProbabilities.copy().multiply(1.0 - rsp)
    return transitionProbabilities.copy().multiply(1.0 - rsp) + alpha

def powerIteration(transitionWeights, papers_h_index, rsp=0.15, epsilon=0.00001, maxIterations=1000):
    if (DEBUG):
        pandas.set_option('display.max_rows', 50)
        pandas.set_option('display.max_columns', 50)
        pandas.set_option('display.width', 1000)    # Clerical work:
    transitionWeights = pandas.DataFrame(transitionWeights)
    nodes = __extractNodes(transitionWeights)
    transitionWeights = __makeSquare(transitionWeights, nodes, default=0.0)
    transitionWeights = __ensureRowsPositive(transitionWeights)

    # Setup:
    state = __startState(nodes, papers_h_index)
    papers_h_index = pandas.Series(papers_h_index)
    transitionProbabilities = __normalizeRows(transitionWeights)
    transitionProbabilities = __integrateRandomSurfer(nodes, transitionProbabilities, rsp, papers_h_index)

    # Power iteration:
    for iteration in range(maxIterations):
        oldState = state.copy()
        state = state.dot(transitionProbabilities)
        delta = state - oldState
        if __euclideanNorm(delta) < epsilon:
            break

    return state

def main():
    nodes = {'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'}
    papers_h_index = {x: 1 for x in nodes}
    edgeWeights = collections.defaultdict(lambda: collections.Counter())
    edgeWeights['p7']['p1'] = 1
    edgeWeights['p7']['p2'] = 1
    edgeWeights['p6']['p3'] = 1
    edgeWeights['p6']['p4'] = 1
    edgeWeights['p8']['p5'] = 1
    edgeWeights['p8']['p6'] = 1
    wordProbabilities = powerIteration(edgeWeights, papers_h_index, rsp=0.5)
    source_norm_probs = (1/16)/(23/32)
    one_level_link_norm_probs = (1/8)/(23/32)
    two_level_link_norm_probs = (5 / 32) / (23 / 32)
    print(wordProbabilities)
    assert (abs(wordProbabilities['p1'] - source_norm_probs) < 0.00001)
    assert (abs(wordProbabilities['p2'] - source_norm_probs) < 0.00001)
    assert (abs(wordProbabilities['p3'] - source_norm_probs) < 0.00001)
    assert (abs(wordProbabilities['p4'] - source_norm_probs) < 0.00001)
    assert (abs(wordProbabilities['p5'] - source_norm_probs) < 0.00001)
    assert (abs(wordProbabilities['p6'] - one_level_link_norm_probs) < 0.00001)
    assert (abs(wordProbabilities['p7'] - one_level_link_norm_probs) < 0.00001)
    assert (abs(wordProbabilities['p8'] - two_level_link_norm_probs) < 0.00001)
    print('test passed')


if __name__ == '__main__':
    main()