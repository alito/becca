import numpy as np

""" Shared constants """
EPSILON = 10e-9
BIG = 10. ** 20

""" Utility functions """
def weighted_average(values, weights):
    weighted_sum_values = np.sum(values * weights, axis=1) 
    sum_of_weights = np.sum(weights, axis=1) 
    return (weighted_sum_values / (sum_of_weights + EPSILON))[:,np.newaxis]


def map_one_to_inf(a):
    """ Map values from [0, 1] onto [0, inf) and map values from [-1, 0] onto (-inf, 0] """
    eps = np.finfo(np.double).eps
    a_prime = np.sign(a) / (1 - np.abs(a) + eps) - np.sign(a)
    return a_prime


def map_inf_to_one(a_prime):
    """ Map values from [0, inf) onto [0, 1] and map values from  (-inf, 0] onto [-1, 0] """
    a = np.sign(a_prime) * (1 - 1 / (np.abs(a_prime) + 1))
    return a
