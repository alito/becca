import sys

import matplotlib.pyplot as plt
import numpy as np

""" 
Constants and functions for used across the BECCA core
"""
# Shared constants
EPSILON = sys.float_info.epsilon
BIG = 10. ** 20
def weighted_average(values, weights):
    """ Perform a weighted average of values, using weights """
    weighted_sum_values = np.sum(values * weights, axis=0) 
    sum_of_weights = np.sum(weights, axis=0) 
    return (weighted_sum_values / (sum_of_weights + EPSILON))[:,np.newaxis]

def map_one_to_inf(a):
    """ ZipTie values from [0, 1] onto [0, inf) and map values 
    from [-1, 0] onto (-inf, 0] """
    eps = np.finfo(np.double).eps
    a_prime = np.sign(a) / (1 - np.abs(a) + eps) - np.sign(a)
    return a_prime

def map_inf_to_one(a_prime):
    """ ZipTie values from [0, inf) onto [0, 1] and map values 
    from  (-inf, 0] onto [-1, 0] """
    a = np.sign(a_prime) * (1 - 1 / (np.abs(a_prime) + 1))
    return a

def bounded_sum(a, axis=0):
    """ 
    Sum elements nonlinearly, such that the total is less than 1 
    
    To be more precise, as long as all elements in a are between -1
    and 1, their sum will also be between -1 and 1. a can be a 
    list or a numpy array. 
    """ 
    if type(a) is list:
        total = map_one_to_inf(a[0])
        for item in a[1:]:
            total += map_one_to_inf(item)
        return map_inf_to_one(total)
    else:
        bounded_total = map_inf_to_one(np.sum(map_one_to_inf(a), axis=axis))
        return bounded_total[:,np.newaxis]

def pad(a, shape, val=0.):
    """
    Pad a numpy array to the specified shape
    
    If any element of shape is <= 0, that size remains unchanged in 
    that axis. Use val (default 0) to fill in the extra spaces. 
    """
    if shape[0] <= 0:
        rows = a.shape[0] - shape[0]
    else:
        rows = shape[0]
        # debug
        if rows < a.shape[0]:
            print ' '.join(['a.shape[0] is', str(a.shape[0]), ' but trying to',
                            ' pad to ', str(rows), 'rows.'])
    if shape[1] <= 0:
        cols = a.shape[1] - shape[1]
    else:
        cols = shape[1]
        # debug
        if cols < a.shape[1]:
            print ' '.join(['a.shape[1] is', str(a.shape[1]), ' but trying to',
                            ' pad to ', str(cols), 'cols.'])
    padded = np.ones((rows,cols)) * val
    padded[:a.shape[0], :a.shape[1]] = a

    return padded

def str_to_int(exp):
    """ Convert a string to an integer """ 
    sum = 0
    for character in exp:
        sum += ord(character)
    return sum

def visualize_array(image_data, shape=None, save_eps=False, 
                    label='data_figure', epsfilename=None):
    """ Produce a visual representation of the image_data matrix """    
    if shape is None:
        shape = image_data.shape
    if epsfilename is None:
        epsfilename = 'log/' + label + '.eps'
    fig = plt.figure(str_to_int(label))
    
    # Diane made the brilliant suggestion to leave this plot in color. 
    # It looks much prettier.
    plt.summer()
    im = plt.imshow(image_data[0:shape[0], 0:shape[1]])
    im.set_interpolation('nearest')
    plt.title(label)
    fig.show()
    fig.canvas.draw()
    if save_eps:
        fig.savefig(epsfilename, format='eps')
    return
