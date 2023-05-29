import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random

# # # # # # # # # # KNN Method project # # # # # # # # #
# # # # # # # Adam Trentowski - 162602 - ISI 2 # # # # #


# Metrics used in function:
def euclidean(p, q):
    """
    Euclidean method of calculating distance

    :param p: point p
    :param q: point q
    :return: distance between p and q using the Euclidean method
    """
    return np.sqrt(np.sum((p - q) ** 2))


def manhattan(p, q):
    """
    Manhattan method of calculating distance

    :param p: point p
    :param q: point q
    :return: distance between p and q using the Manhattan method
    """
    return np.sum(np.abs(p - q))


def cosine(p, q):
    """
    Cosine method of calculating distance

    :param p: point p
    :param q: point q
    :return: distance between p and q using the Cosine similarity method
    """
    fraction_top = np.dot(p, q)
    fraction_bottom = np.linalg.norm(p) * np.linalg.norm(q)
    return 1 - (fraction_top / fraction_bottom)


# kNN function
def knn(x_train, y_train, x_test, k=3, metric=euclidean, weighted=False):
    """
    k-Nearest Neighbors (kNN) algorithm

    :param x_train: training set containing attributes of objects
    :param y_train: labels corresponding to the training data in x_train
    :param x_test: test set containing attributes of objects for which kNN predict the result
    :param k: number of nearest neighbors to consider when deciding (default 3)
    :param metric: metric function used to calculate the distance between attributes (default euclidean method)
    :param weighted: specifies whether to consider weighting when making decisions (default false)
    :return: list of predicted labels for objects in x_test
    """
    # returned list of predicated labels for objects in x_test
    y_predict = []
    # loop iterating through all elements of test set
    for test_element in x_test:
        # list of distances of the test element to the elements of the training set
        distances = []
        # loop iterating through all elements of training set
        for train_element in x_train:
            # calculating distance from test_element to train_element using function passed as argument
            distance = metric(test_element, train_element)
            # adding distance to list
            distances.append(distance)

        # indexes list of distances list sorted in ascending order
        sorted_indexes = np.argsort(distances)
        k_nearest_indices = sorted_indexes[:k]
        k_nearest_labels = y_train[k_nearest_indices]

        # if a weighted classifier is to be used
        if weighted:
            # initialize list of weights (calculating weights for k-nearest elements)
            weights = np.empty(len(k_nearest_indices))
            # loop iterating through all nearest distances (enumerate return index + value)
            for i, index in enumerate(k_nearest_indices):
                d = distances[index]
                # to avoid division by 0
                if distances[index] == 0:
                    d = 0.000001
                weights[i] = 1 / d

            # initialize the list of occurrences of each label in its nearest neighbors [including weights]
            # e.g. if there are labels [0, 2, 1, 0], labels_counts will initialize as [0, 0 ,0]
            label_counts = {}
            # loop iterating through all nearest labels (enumerate return index + value)
            for i, label in enumerate(k_nearest_labels):
                # adding weights
                if label in label_counts:
                    label_counts[label] += weights[i]
                else:
                    label_counts[label] = weights[i]

        # if a weighted classifier is not to be used
        else:
            # initialize the list of occurrences of each label in its nearest neighbors
            label_counts = {}
            # iterate through all nearest labels and count them
            for label in k_nearest_labels:
                # adding number of occurrences
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

        # select label with the most occurrences as the predicted label
        predicted_label = max(label_counts, key=label_counts.get)

        y_predict.append(predicted_label)

    return y_predict


# Method to measure accuracy
def accuracy(set_1, set_2):
    return (np.sum(set_1 == set_2) / len(set_1)) * 100


# Example of use:

# The Iris dataset contains information on three iris species:
# Setosa, Versicolor and Virginica. Each iris is represented by four characteristics:
#
#      Sepal length - measured in centimeters.
#      Sepal width - measured in centimeters.
#      Petal length - measured in centimeters.
#      Petal width - measured in centimeters.
#
# The data for each iris consists of these four numeric values.
# In addition to the input data, the dataset also contains information
# about the classes of belonging of individual irises to one of three species.
#
# In total, the Iris dataset consists of 150 examples of irises,  with 50 examples for each of the three species.

iris = load_iris()

# generating random number
seed = random.randint(1, 100)
print("\n                       SEED: " + str(seed) + "\n")
# data division into training and test sets
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=seed)

# # # # # # # # # # # # # # # # # # # # # # # # EUCLIDEAN # # # # # # # # # # # # # # # # # # # # # # # #
result_1 = knn(x_train, y_train,  x_test)
accuracy_1 = accuracy(y_test, result_1)
print("Euclidean accuracy - k=3 [non weighted]:", accuracy_1)

result_2 = knn(x_train, y_train,  x_test, weighted=True)
accuracy_2 = accuracy(y_test, result_2)
print("Euclidean accuracy - k=3 [weighted]    :", accuracy_2)

result_3 = knn(x_train, y_train,  x_test, k=5)
accuracy_3 = accuracy(y_test, result_3)
print("Euclidean accuracy - k=5 [non weighted]:", accuracy_3)

result_4 = knn(x_train, y_train,  x_test, k=5, weighted=True)
accuracy_4 = accuracy(y_test, result_4)
print("Euclidean accuracy - k=5 [weighted]    :", accuracy_3)

# # # # # # # # # # # # # # # # # # # # # # # # MANHATTAN # # # # # # # # # # # # # # # # # # # # # # # #
result_5 = knn(x_train, y_train,  x_test, k=3, metric=manhattan, weighted=False)
accuracy_5 = accuracy(y_test, result_5)
print("\n\nManhattan accuracy - k=3 [non weighted]:", accuracy_5)

result_6 = knn(x_train, y_train,  x_test, k=3, metric=manhattan, weighted=True)
accuracy_6 = accuracy(y_test, result_6)
print("Manhattan accuracy - k=3 [weighted]    :", accuracy_6)

result_7 = knn(x_train, y_train,  x_test, k=5, metric=manhattan, weighted=False)
accuracy_7 = accuracy(y_test, result_7)
print("Manhattan accuracy - k=5 [non weighted]:", accuracy_7)

result_8 = knn(x_train, y_train,  x_test, k=5, metric=manhattan, weighted=True)
accuracy_8 = accuracy(y_test, result_8)
print("Manhattan accuracy - k=5 [weighted]    :", accuracy_8)

# # # # # # # # # # # # # # # # # # # # # # # # COSINE # # # # # # # # # # # # # # # # # # # # # # # #
result_9 = knn(x_train, y_train,  x_test, k=3, metric=cosine, weighted=False)
accuracy_9 = accuracy(y_test, result_9)
print("\n\nCosine accuracy - k=3 [non weighted]:", accuracy_9)

result_10 = knn(x_train, y_train,  x_test, k=3, metric=cosine, weighted=True)
accuracy_10 = accuracy(y_test, result_10)
print("Cosine accuracy - k=3 [weighted]    :", accuracy_10)

result_11 = knn(x_train, y_train,  x_test, k=5, metric=cosine, weighted=False)
accuracy_11 = accuracy(y_test, result_11)
print("Cosine accuracy - k=5 [non weighted]:", accuracy_11)

result_12 = knn(x_train, y_train,  x_test, k=5, metric=cosine, weighted=True)
accuracy_12 = accuracy(y_test, result_12)
print("Cosine accuracy - k=5 [weighted]    :", accuracy_12)
