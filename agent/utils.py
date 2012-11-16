import numpy as np

""" Shared constants """
EPSILON = 10e-9

""" Utility functions """

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
            
        if a.shape != b.shape:
            raise ValueError("Both parameters have to have the same shape." \
                              " Got %s and %s" % (a.shape, b.shape))

        if a.size == 0:
            return np.zeros(a.shape)
        
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
    a_prime = map_one_to_inf(a[same_sign_indices])
    b_prime = map_one_to_inf(b[same_sign_indices])
    c_same_sign = map_inf_to_one(a_prime + b_prime) 
    
    """ Compile the results """
    result[same_sign_indices] = c_same_sign
    result[opposite_sign_indices] = c_opposite_sign

    if scalars:
        """ Return the same type that we started with """
        return result[0]
    else:
        return result
    
    
def map_one_to_inf(a):
    """ Map values from [0, 1] onto [0, inf) and 
    map values from [-1, 0] onto (-inf, 0].
    """
    eps = np.finfo(np.double).eps
    a_prime = np.sign(a) / (1 - np.abs(a) + eps) - np.sign(a)
    return a_prime


def map_inf_to_one(a_prime):
    """ Map values from [0, inf) onto [0, 1] and 
    map values from  (-inf, 0] onto [-1, 0].
    """
    a = np.sign(a_prime) * (1 - 1 / (np.abs(a_prime) + 1))
    return a


def similarity(point, point_set, max_index=None):
    
    """
    Calculate the similarity between a point (an array) and a set of points.
    point_set must be a 2D 
    numpy array with a zeroth dimension of the same length.
    
    Return a vector containing the similarity between each point 
    in point_set[:max_index] and point. 

    The angle between the vectors was chosen as the basis
    for the similarity measure:
    
    s(a,b) = 1 - theta / (pi/2) 
    
    It has several desirable properties:
    If the similarity between two points is s(a,b)
    1) s(a,b) = s(b,a), transitive
    2) s(a,b) is on [0,1]
    3) if a and b share no nonzero elements, s(a,b) = 0;
    4) s(a,b) = 1 iff a = b * c, where c is a constant > 0
    """
    
    """ Check to see whether either input is empty """ 
    if not (point_set.size > 0) or not (point.size > 0):
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

    """ Make sure point is a 2D numpy column array """
    if len(point.shape) == 1:
        point = point[:,np.newaxis]
    if point.shape[0] == 1:
        point = point.transpose()
            
    if point.shape[0] != point_set.shape[0]:
        print "Error in utils.similarity(): point must have the same number"
        print "elements as the 0th dimension of point_set. "
        print "Got ", point.shape[0] ,' and ', point_set.shape[0]
        raise ValueError
    
    """ Expand the point array to a 2D array the same size 
    as the point_set.
    """
    point_mat = np.tile(point, (1, max_index))
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
        
    return result
