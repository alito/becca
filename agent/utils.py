"""
Utility functions
"""

import logging
import copy

import numpy as np

class AutomaticList(list):
    """
    A list-like class that behaves like Matlab's arrays in automatically extending when an index beyond
    its length is set
    """
    def __setitem__(self, key, value):
        missing = key - len(self)
        if missing < 0:
            list.__setitem__(self, key, value)
        else:
            # set missing items to a null array
            for i in range(missing):
                self.append(empty_array())
            self.append(value)
                
            

def bounded_sum(a, b):
    """
    Sums the values A and B, which are assumed to fall on the interval
    [-1,1], to create a value C with the following properties:
    
    for A and B on [0,1]
    1. if C = bsum(A,B), C > A, C > B
    2. C is on [0,1]
    3. bsum(A,B) = bsum(B,A)
    4. if A1 > A2, bsum(A1,B) > bsum(A2, B)
    
    opposites are true for A and B on [-1,0]
    
    for A and B on [-1,1]
    1. C is on [-1,1]
    
    for A and B oppositely signed, C = A + B
    
    if A and B are vectors or matrices, they must be of the same size, and
    C will be of the same size too


    """

    eps = np.finfo(np.double).eps
    
    result = np.zeros(np.shape(a))
    # maps [0,1]  onto [1,Inf)
    # maps [-1,1] onto (-Inf,Inf)
    
    # different functions for when a and b are same or opposite signs
    same_sign_indices = (np.sign(a) == np.sign(b)).ravel().nonzero()
    
    a_ss = a[same_sign_indices]
    b_ss = b[same_sign_indices]
    a_t = np.sign(a_ss) / (1 - np.abs(a_ss) + eps) - np.sign(a_ss)
    b_t = np.sign(b_ss) / (1 - np.abs(b_ss) + eps) - np.sign(b_ss)
    
    c_t = a_t + b_t 
    
    c_ss = np.sign(c_t) * (1 - 1 / (np.abs(c_t) + 1))
    c_ss[ c_t == 0] = 0
    
    opposite_sign_indices = (np.sign(a) != np.sign(b)).ravel().nonzero()
    a_os = a[opposite_sign_indices]
    b_os = b[opposite_sign_indices]
    c_os = a_os + b_os
    
    result[same_sign_indices] = c_ss
    result[opposite_sign_indices] = c_os

    return result



def similarity(point, set_of_points, indices):
    """
    Calculates the similarity between a point and a set of points.
    Returns a vector with the similarity between each point in set_of_points[indices]
    from the initial POINT.

    The angle between the vectors was chosen as the basis
    for the similarity measure:
    
    s(a,b) = 1 - theta / (pi/2) 
    
    It has several desirable properties:
    If the similarity between two points is s(a,b)
    1) s(a,b) = s(b,a), transitive
    2) s(a,b) is on [0,1] if all elements of a and b are on [0,1]
    3) if a and b share no nonzero elements, s(a,b) = 0;
    4) s(a,b) = 1 iff a = b * c, where c is a constant > 0
    """
    
    result = None

    if not indices or not set_of_points or not point:
        return None

    eps = np.finfo(np.double).eps    
    num_points = len(indices)

    # first handles the non-cell case, e.g. comparing inputs to 
    # the features within a group
    if point.dtype.name != 'object':
        
        point_mat = np.tile(point, (1, num_points))
        set_mat = set_of_points[:,indices]
        inner_product = np.sum(( point_mat * set_mat), axis=0)
        mag_point = np.sqrt(np.sum( point_mat ** 2, axis=0)) + eps
        mag_set = np.sqrt(np.sum(set_mat ** 2, axis=0)) + eps
        cos_theta = inner_product / (mag_point * mag_set)

        cos_theta = np.minimum(cos_theta, 1)
        
        theta = np.arccos( cos_theta)
        result = 1 - theta / ( np.pi/2)

    # the handles the cell case, e.g. comparing full feature activities
    # against causes in the model
    else:
        num_groups = len(point)
        inner_product = np.zeros(num_points)
        sum_sq_point = np.zeros(num_points)
        sum_sq_set = np.zeros(num_points)

        for index in range(0,num_groups):
            if point[index]:

                # this weighting factor weights matches more heavily in groups
                # with more features. In a group with one feature, a match
                # means less than in a group with 10 features.
                # TODO: consider whether to continue using the weighting
                weight = len(point[index])
                point_mat = np.tile(point[index], (1, num_points))
                set_mat = set_of_points[index][:,indices]
                inner_product += weight * np.sum((point_mat * set_mat), axis=0)
                sum_sq_point += weight * np.sum(point_mat ** 2, axis=0)
                sum_sq_set += weight * np.sum(set_mat ** 2, axis=0)

                
        mag_point = np.sqrt( sum_sq_point) + eps
        mag_set = np.sqrt( sum_sq_set) + eps
        cos_theta = inner_product / (mag_point * mag_set)

        cos_theta = np.minimum(cos_theta, 1)
        # the 'real' was added to compensate for a numerical processing
        # artifact resulting in a cos_theta slightly greater than 1
        theta = np.arccos(cos_theta)
        result = 1 - theta / ( np.pi/2)

    return result



def sigmoid(a):
    return (2 / (1 + np.exp(-2 * a))) - 1


def empty_array():
    return np.zeros((0,0))

def winner_takes_all(feature_input):
    
    # performs winner-take-all on primed_vote to get feature_output
    #
    # no WTA is performed on group 1, the raw sensory group, group 2, the 
    # hard-wired feature group, or group 3, motor group.  
    # All motor commands pass through to become feature outputs.
    num_groups = len(feature_input)

    feature_output = []
    feature_output.append(empty_array())
    
    feature_output.append(copy.deepcopy(feature_input[1]))
    feature_output.append(copy.deepcopy(feature_input[2]))
    
    for index in range(3, num_groups):
        max_index = np.argmax( np.abs(feature_input[index]))
        feature_output[index] = np.zeros( feature_input[index].shape[0])
        feature_output[index][max_index] = feature_input[index][max_index]

    return feature_output

