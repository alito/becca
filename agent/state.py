import copy
import numpy as np
import utils

class State(object):
    """ A data structure for representing the internal state of the agent """ 

    def __init__(self, 
                 num_primitives=0, 
                 num_actions=0, 
                 num_total_features=0,
                 width=1):
        """ Constructor from scratch """
        self.num_primitives = num_primitives
        self.num_actions = num_actions
        """ Handle the case where num_total_features is not assinged """
        if num_total_features == 0:
            num_total_features = num_primitives + num_actions
            
        self.features = np.zeros((num_total_features, width))
        
        
    def zeros_like(self, dtype=np.float):
        """  Create a new state instance the same size as old_state, 
        but all zeros
        """
        state_type = dtype
        zero_state = copy.deepcopy(self)
        zero_state.features = np.zeros_like(self.features, dtype=state_type)
        
        return zero_state
        
        
    def ones_like(self, dtype=np.float):
        """  Create a new state instance the same size as old_state, 
        but all ones
        """
        state_type = dtype
        ones_state = copy.deepcopy(self)
        ones_state.features = np.ones_like(self.features, dtype=state_type)
        
        return ones_state
        
        
    def set_primitives(self, primitives):
        if primitives.size != self.num_primitives:
            print('You tried to assign the wrong number of primitives to a state.')
        else: 
            self.features[0:self.num_primitives, 0] = \
                            np.copy(primitives).ravel()
        return

    
    def get_primitives(self):
        return (np.copy(self.features[:self.num_primitives, :]))
        
        
    def set_actions(self, actions):
        if actions.size != self.num_actions:
            print('You tried to assign the wrong number of actions to a state.')
        else:
            if actions.ndim == 1:
                actions = actions[:,np.newaxis]
            self.features[self.num_primitives:
                  self.num_primitives + self.num_actions, 0] = \
                                  np.copy(actions).ravel()
    
    
    def get_actions(self):
        return (np.copy(self.features[self.num_primitives:
                  self.num_primitives + self.num_actions,:]))
        
        
    def set_features(self, features):
        if features.ndim == 1:
            features = features[:,np.newaxis]
        self.features = np.copy(features)
        
        
    def is_shaped_like(self, other_state):
        is_like =True
        
        if self.num_primitives != other_state.num_primitives:
            is_like = False
            return is_like
            
        if self.num_actions != other_state.num_actions:
            is_like = False
            return is_like

            
        if self.hi_element() != other_state.hi_element():
            is_like = False
            return is_like
            
        return is_like
    
    
    def equals(self, other_state):
        """ Test whether two states are in practice equal """
        is_equal = True
        
        is_like = self.is_shaped_like(other_state)
        
        if not is_like:
            is_equal = False
            return is_equal
        
        if self.hi_element() > 0:
            if np.max(self.features[:self.hi_element()] - 
                  other_state.features[:self.hi_element()]) > utils.EPSILON:
                is_equal = False
                return is_equal
        
        return is_equal
    
    
    def is_close(self, other_state, similarity_threshold):
        """ Test whether two states are similar to each other """
        is_close = True
        
        is_like = self.is_shaped_like(other_state)
        if not is_like:
            is_close = False
            return is_close
        
        if self.hi_element() > 0:
            if utils.similarity(self.features[:self.hi_element()], 
                                other_state.features[:self.hi_element()]) \
                                > similarity_threshold:
                is_close = False
                return is_close
            
        return is_close
        
        
    def unbounded_sum(self, other_state):
        """ Add another State to this State. 
        Values of individual features may have a magnitude greater than 1.
        """
        new_state = other_state.zeros_like()        
        new_state.features = self.features + other_state.features
                
        return new_state
        

    def bounded_sum(self, other_state):
        """ Add another State to this State, 
        ensuring that no value has a magnitude greater than one """
        new_state = other_state.zeros_like()        
        
        """ Check whether the two states are the same size and 
        pad them if not.
        """
        min_size = np.minimum(new_state.features.size, self.n_features())
        max_size = np.maximum(new_state.features.size, self.n_features())
        
        new_features = utils.bounded_sum(self.features[:min_size,:], 
                                         other_state.features[:min_size,:])
        padded_features = np.zeros((max_size,1))
        padded_features[:min_size,:] = new_features
        new_state.set_features(padded_features)

        return new_state

    
    def multiply(self, multiplier):
        """ Multiply this State by a scalar """
        self.features *= multiplier
        return

    
    def integrate_state(self, new_state, decay_rate):
        """ Updates the state by combining the new state value with a 
        decayed version of the current state value.
        """
    
        integrated_state = new_state.zeros_like()
        integrated_state.features = utils.bounded_sum(
                                self.features * (1 - decay_rate), 
                                new_state.features)
                    
        return integrated_state
            
            
    def prepend(self, array):
        """ Return the entire state with array tacked on 
        to the beginning of it. 
        """ 
        return(np.concatenate((array, copy.deepcopy(self.features))))
        
        
    def n_features(self):
        return self.features.size
    
    
    def hi_element(self):
        """ Return the index of the largest significantly non-zero feature """
        nonzero_indices = self.features.ravel().nonzero()[0]
        if nonzero_indices.size != 0:
            hi_element = np.max(nonzero_indices)
        else:
            hi_element = 0
            
        return hi_element
        