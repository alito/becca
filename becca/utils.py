"""
Utility functions
"""

import logging
import copy

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

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
    scalars = np.isscalar(a)
    if scalars:
        if not np.isscalar(b):
            raise ValueError("both parameters have to be of the same type")

        a = np.array([a])
        b = np.array([b])
    else:
        if a.shape != b.shape:
            raise ValueError("Both parameters have to have the same shape. Got %s and %s" % (a.shape, b.shape))
    

    
    eps = np.finfo(np.double).eps
    
    result = np.zeros(np.shape(a))
    # maps [0,1]  onto [1,Inf)
    # maps [-1,1] onto (-Inf,Inf)
    
    # different functions for when a and b are same or opposite signs
    same_sign_indices = (np.sign(a) == np.sign(b)).ravel().nonzero()
    
    a_same_sign = a[same_sign_indices]
    b_same_sign = b[same_sign_indices]
    a_t = np.sign(a_same_sign) / (1 - np.abs(a_same_sign) + eps) - np.sign(a_same_sign)
    b_t = np.sign(b_same_sign) / (1 - np.abs(b_same_sign) + eps) - np.sign(b_same_sign)
    
    c_t = a_t + b_t 
    
    c_same_sign = np.sign(c_t) * (1 - 1 / (np.abs(c_t) + 1))
    c_same_sign[ c_t == 0] = 0
    
    opposite_sign_indices = (np.sign(a) != np.sign(b)).ravel().nonzero()
    a_opposite_sign = a[opposite_sign_indices]
    b_opposite_sign = b[opposite_sign_indices]
    c_opposite_sign = a_opposite_sign + b_opposite_sign
    
    result[same_sign_indices] = c_same_sign
    result[opposite_sign_indices] = c_opposite_sign

    if scalars:
        # return same type that we got
        return result[0]
    else:
        return result



def similarity(point, set_of_points, max_index=None):
    """
    Calculates the similarity between a point and a set of points.
    Returns a vector with the similarity between each point in set_of_points[:max_index]
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

    if not np.size(set_of_points) or not np.size(point):
        return None

    eps = np.finfo(np.double).eps

    if max_index is None:
        # if there is no maximum index, set it to the length of the set
        max_index = set_of_points.shape[1]

    # first handles the non-cell case, e.g. comparing inputs to 
    # the features within a group
    if not isinstance(point, list):

        # make a point matrix of the same size as the set matrix
        # need to convert the point vector to a matrix
        point_mat = np.tile(point[np.newaxis].transpose(), (1, max_index))
        set_mat = set_of_points[:,:max_index]
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
        inner_product = np.zeros(max_index)
        sum_sq_point = np.zeros(max_index)
        sum_sq_set = np.zeros(max_index)

        for index in range(num_groups):
            if np.size(point[index]):
                # this weighting factor weights matches more heavily in groups
                # with more features. In a group with one feature, a match
                # means less than in a group with 10 features.
                # TODO: consider whether to continue using the weighting
                weight = len(point[index])
                point_mat = np.tile(point[index][np.newaxis].transpose(), (1, max_index))
                set_mat = set_of_points[index][:,:max_index]
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
        feature_output.append(np.zeros( feature_input[index].shape[0]))
        feature_output[index][max_index] = feature_input[index][max_index]

    return feature_output



def force_redraw():
    """
    Force matplotlib to draw things on the screen
    """
    
    # pause is needed for events to be processed
    # Qt backend needs two event rounds to process screen. Any number > 0.01 and <=0.02 would do
    plt.pause(0.015)
