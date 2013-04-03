import numpy as np

""" Shared constants """
EPSILON = 10e-9
BIG = 10. ** 20

""" Utility functions """
def weighted_average(values, weights):
    weighted_sum_values = np.sum(values * weights, axis=0) 
    sum_of_weights = np.sum(weights, axis=0) 
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

def bounded_sum(a, axis=0):
    if type(a) is list:
        total = map_one_to_inf(a[0])
        for item in a[1:]:
            total += map_one_to_inf(item)
        return map_inf_to_one(total)
    # else
    bounded_total = map_inf_to_one(np.sum(map_one_to_inf(a), axis=axis))
    #if axis==0:
    #    return bounded_total[np.newaxis,:]        
    #else
    return bounded_total[:,np.newaxis]

def pad(a, shape, val=0.):
    if shape[0] <= 0:
        rows = a.shape[0] - shape[0]
    else:
        rows = shape[0]
    if shape[1] <= 0:
        cols = a.shape[1] - shape[1]
    else:
        cols = shape[1]
    padded = np.ones((rows,cols)) * val
    padded[:a.shape[0], :a.shape[1]] = a
    return padded

def ord_str(label):
    sum = 0
    for character in label:
        sum += ord(character)
    return sum
