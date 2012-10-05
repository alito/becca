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
            #if primitives.ndim == 1:
            #        primitives = primitives[:,np.newaxis]
            self.features[0:self.num_primitives, 0] = primitives.ravel()
        
        
    def set_actions(self, actions):
        if actions.size != self.num_actions:
            print('You tried to assign the wrong number of actions to a state.')
        else:
            #if actions.ndim == 1:
            #        actions = actions[:,np.newaxis]
            self.features[self.num_primitives:
                  self.num_primitives + self.num_actions, 0] = actions.ravel()
    
    
    def get_actions(self):
        return (self.features[self.num_primitives:
                  self.num_primitives + self.num_actions,:])
        
        
    def set_features(self, features):
        if features.ndim == 1:
            features = features[:,np.newaxis]
            
        self.features = features
        
        
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
        new_state = self.zeros_like()
        new_state.features = self.features * multiplier
            
        return new_state

    
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
        