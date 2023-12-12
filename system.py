"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np

from scipy import linalg


N_DIMENSIONS = 10


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This implementation uses a k-NN classifier, where k = 5.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    
    K_NEIGHBOURS = 10
    results = []

    #Calculate distance measure matrix - Euclidean distance is implemented here
    train_squared = np.sum(np.square(train), axis=1)
    test_squared = np.sum(np.square(test), axis=1)
    dot_product = np.dot(test, train.T)
    distances = np.sqrt(test_squared[:, None] - 2 * dot_product + train_squared)
    
    for i in range(distances.shape[0]):
        #Find k-nearest neighbours
        nearest_neighbours = np.argsort(distances[i])[:K_NEIGHBOURS]

        #Calculate the class
        classes = train_labels[nearest_neighbours]
        unique_classes, counts = np.unique(classes, return_counts=True)
        max_position = np.argmax(counts)

        results.append(unique_classes[max_position])

    return results


# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    This implementation uses Principal Component Analysis to reduce the dimensions to 10.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """

    N = 10 #Number of dimensions

    mean = np.array(np.mean(data, axis=0))
    normalised_data = (data - mean)
    cov_matrix = np.cov(normalised_data, rowvar=False)
    size = cov_matrix.shape[0]

    eigenvalues, eigenvectors = linalg.eigh(cov_matrix, subset_by_index=[size-N, size-1])

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = np.array(eigenvectors[:, sorted_indices])

    pca_data = np.dot(normalised_data, eigenvectors)

    #The mean and eigenvectors of the data are stored to use when reconstructing the data
    model["eigenvectors"] = eigenvectors.tolist()
    model["mean"] = mean.tolist()

    return pca_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Stores the labels of the training data, the reduced training data, the eigenvectors of 
    the training data and the mean of the training data in the model for use in reconstructing
    the data and classifying test data.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.
        eigenvectors_train (np.ndarray): the Principal Components of the feature vectors.
        mean_train (np.ndarray): the mean of the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    model = {}

    model["labels_train"] = labels_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()

    model["eigenvectors_train"] = model["eigenvectors"]
    model["mean_train"] = model["mean"]

    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Takes all the necessary data from the model, reconstructs the reduced data and runs the 
    classifier on the array of images (feature vectors).

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    train_vectors = np.array(model["eigenvectors_train"])
    train_mean = np.array(model["mean_train"])

    test_vectors = np.array(model["eigenvectors"])
    test_mean = np.array(model["mean"])

    #Reconstructs the reduced data 
    train_data = np.array(np.dot(fvectors_train, train_vectors.T) + train_mean)
    test_data = np.array(np.dot(fvectors_test, test_vectors.T) + test_mean)

    #Classify the data
    labels = classify(train_data, labels_train, test_data)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    Takes all the necessary data from the model and reconstructs the reduced data. 
    It then splits the training data, labels and test data into black or white squares
    and classifies them independently. 

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    train_vectors = np.array(model["eigenvectors_train"])
    train_mean = np.array(model["mean_train"])

    test_vectors = np.array(model["eigenvectors"])
    test_mean = np.array(model["mean"])

    #Reconstructs the reduced data 
    train_data = np.dot(fvectors_train, train_vectors.T) + train_mean
    test_data = np.dot(fvectors_test, test_vectors.T) + test_mean

    #Split the training data and labels into black and white squares
    white_squares = np.array(train_data[0::2,:])
    white_labels = np.array(labels_train[0::2])

    black_squares = np.array(train_data[1::2,:])
    black_labels = np.array(labels_train[1::2])

    #Split the test data into black and white squares
    white_tests = np.array(test_data[0::2,:])
    black_tests = np.array(test_data[1::2,:])

    #Classify the black and white squares independently
    white_results = classify(white_squares, white_labels, white_tests)
    black_results = classify(black_squares, black_labels, black_tests)

    results = [None] * (len(white_results) + len(black_results))
    results[0::2],results[1::2] = white_results, black_results

    return results

    