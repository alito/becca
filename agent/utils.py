import numpy as np
import state

""" Utility functions """

'''  not used now?
class AutomaticList(list):
    """ A list-like class that behaves like Matlab's arrays in 
    automatically extending when an index beyond
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
'''               

def bounded_sum(a, b):
    """ Perform a bounded sum.
    Sum the values A and B, which are assumed to fall on the interval
    [-1,1], to create a value C with the following properties:
    
    For A and B on [0,1] and
    C = bsum(A,B)
    1. C > A and C > B
    2. C is on [0,1]
    3. bsum(A,B) = bsum(B,A)
    4. if A1 > A2, bsum(A1,B) > bsum(A2, B)
    
    The opposite of statesments 1-4 are true for A and B on [-1,0]
    
    For A and B on [-1,1]
    1. C is on [-1,1]
    
    If A and B are vectors or matrices, they must be of the same size, and
    C will be of the that size too.
    """
    
    """ The functions used to do this are different depending on 
    whether A and B have same or opposite signs.
    
    For A, B opposite signs,
    bsum(A,B) = A + B
    
    For A > 0, B > 0, both A and B are mapped to from the interval [0,1] onto
    the interval [0, infinity) using a transformation, T(), given by
    T(A) = A' = -1 + 1/(1 - A)
    
    The inverse transformation, to remap the variable from 
    the interval [0, infinity) back to [0,1], T'(), is given by
    T'(A') = A = 1 - 1/(1 + A)
    
    Using this notation,  
    bsum(A,B) = T'(T(A) + T(B))
    
    For A < 0, B < 0, 
    bsum(A,B) = -T'(T(-A) + T(-B))
    """
    
    scalars = np.isscalar(a)
    if scalars:
        """ Handle the case where a and b are scalars """
        if not np.isscalar(b):
            raise ValueError("Both parameters have to be of the same type")

        a = np.array([a])
        b = np.array([b])
    else:
        """ Perform error checking on the size of and b """
        if a.size != b.size:
            raise ValueError("Both parameters must have the same number " \
                             "of elements. Got %s and %s" % (a.size, b.size))
            
        if a.size == 0 or b.size == 0:
            return np.zeros((0,0))
        
        if a.shape != b.shape:
            raise ValueError("Both parameters have to have the same shape." \
                              " Got %s and %s" % (a.shape, b.shape))
        
    result = np.zeros(np.shape(a))

    
    """ There are different functions, depending on whether 
    a and b are same or opposite signs. 
    """

    """ First handle the case where a and b are of opposite signs """ 
    opposite_sign_indices = (np.sign(a) != np.sign(b)).nonzero()
    c_opposite_sign = a[opposite_sign_indices] + b[opposite_sign_indices]
    
    """ Then handle the case where a and b are of the same sign """ 
    same_sign_indices = (np.sign(a) == np.sign(b)).nonzero()
    
    """ Map [0,1]  onto [0,Inf) and  maps [-1,0] onto (-Inf,0] 
    Then map back after dum is completed.
    """
    a_prime = bounded_sum_map(a[same_sign_indices])
    b_prime = bounded_sum_map(b[same_sign_indices])
    c_same_sign = bounded_sum_unmap(a_prime + b_prime) 
    
    """ Compile the results """
    result[same_sign_indices] = c_same_sign
    result[opposite_sign_indices] = c_opposite_sign

    if scalars:
        """ Return the same type that we started with """
        return result[0]
    else:
        return result
    
    
def bounded_sum_map(a):
    eps = np.finfo(np.double).eps
    a_prime = np.sign(a) / (1 - np.abs(a) + eps) - np.sign(a)
    return a_prime


def bounded_sum_unmap(a_prime):
    a = np.sign(a_prime) * (1 - 1 / (np.abs(a_prime) + 1))
    return a


def similarity(point, point_set, max_index=None):
    """
    Calculates the similarity between a point and a set of points.
    The variable 'point' may either be a numpy array, or a state. 
    If point is an array, point_set must be a 2D 
    numpy array with a zeroth dimension of the same length.
    If point is a state, point_set must be a 2D state with 
    each group of the same size as state, such
    as the context, cause, and effect variables in the model.
    
    Return a vector containing the similarity between each point 
    in point_set[:max_index] and point. 

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
    
    """ Check to see whether either input is empty """ 
    if not (np.size(point_set) > 0) or not (np.size(point) > 0):
        print "Warning: utils.similarity()-inputs must " + \
              "both be of non-zero size. "
        print "    point_set:"
        print point_set
        print "    point:"
        print point
        return None

    eps = np.finfo(np.double).eps

    """ If there is no maximum group_index, set it to the length of the set """
    if max_index is None:
        max_index = point_set.shape[1]

    if not isinstance(point, state.State):
        """ First handle the non-list case, comparing 
        an array to each column in a 2D array.
        """
        
        """ Expand the point array to a 2D array the same size 
        as the point_set.
        """
        point_mat = np.tile(point[np.newaxis].transpose(), (1, max_index))
        set_mat = point_set[:,:max_index]
        
        """ Calculate the angle between each of the corresponding 
        columns.
        """
        inner_product = np.sum(( point_mat * set_mat), axis=0)
        mag_point = np.sqrt(np.sum( point_mat ** 2, axis=0)) + eps
        mag_set = np.sqrt(np.sum(set_mat ** 2, axis=0)) + eps
        cos_theta = inner_product / (mag_point * mag_set)
        cos_theta = np.minimum(cos_theta, 1)
        theta = np.arccos( cos_theta)
        result = 1 - theta / ( np.pi/2)

    else:
        """ Handle the state case, for example comparing full 
        feature activities to contexts in the model.
        """
        num_groups = point.n_feature_groups()
        inner_product = np.zeros(max_index)
        sum_sq_point = np.zeros(max_index)
        sum_sq_set = np.zeros(max_index)

        """ First take care of primitives """
        if point.primitives.size > 0:
            """ This weighting factor weights matches more heavily in groups
            with more features. In a group with one feature, a match
            means less than in a group with 10 features.
            """
            """ TODO: consider whether to continue using the weighting """
            weight = point.primitives.size
            point_mat = np.tile(point.primitives[np.newaxis].transpose(), 
                                (1, max_index))
            set_mat = point_set.primitives[:,:max_index]
            inner_product += weight * np.sum((point_mat * set_mat), axis=0)
            sum_sq_point += weight * np.sum(point_mat ** 2, axis=0)
            sum_sq_set += weight * np.sum(set_mat ** 2, axis=0)

        """ Second take care of actions in the same way """
        if point.actions.size > 0:
            weight = point.actions.size
            point_mat = np.tile(point.actions[np.newaxis].transpose(), 
                                (1, max_index))
            set_mat = point_set.actions[:,:max_index]
            inner_product += weight * np.sum((point_mat * set_mat), axis=0)
            sum_sq_point += weight * np.sum(point_mat ** 2, axis=0)
            sum_sq_set += weight * np.sum(set_mat ** 2, axis=0)

        """ Finally take care of all the feature groups in the same way """
        for group_index in range(num_groups):
            if point.features[group_index].size > 0:
                weight = point.features[group_index].size
                point_mat = np.tile(point.features[group_index]
                                    [np.newaxis].transpose(), 
                                    (1, max_index))
                set_mat = point_set.features[group_index][:,:max_index]
                inner_product += weight * np.sum((point_mat * set_mat), axis=0)
                sum_sq_point += weight * np.sum(point_mat ** 2, axis=0)
                sum_sq_set += weight * np.sum(set_mat ** 2, axis=0)

        """ Combine the results to calculate the angle 
        between each of the corresponding states.
        """
        mag_point = np.sqrt( sum_sq_point) + eps
        mag_set = np.sqrt( sum_sq_set) + eps
        cos_theta = inner_product / (mag_point * mag_set)
        cos_theta = np.minimum(cos_theta, 1)
        theta = np.arccos(cos_theta)
        result = 1 - theta / ( np.pi/2)

    return result


def sigmoid(a):
    return (2 / (1 + np.exp(-2 * a))) - 1


def empty_array():
    return np.zeros((0,0))


def winner_takes_all(a):
    """ Perform winner-take-all on a """
    max_index = np.argmax(np.abs(a))
    
    """ Enforce a max absolute value of 1.0 """
    if a[max_index] > 1.0:
        a[max_index] = 1.0
    if a[max_index] < -1.0:
        a[max_index] = -1.0
        
    wta_features = np.zeros(a.shape)
    wta_features[max_index] = a[max_index]

    return wta_features


