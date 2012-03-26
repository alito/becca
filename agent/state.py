import copy
import numpy as np
import utils

class State(object):
    """ A data structure for representing the internal state of the agent
    """ 
    sensors = np.array([])
    primitives = np.array([])
    actions = np.array([])
    features = [np.array([])]

    def __init__(self, num_sensors, num_primitives, num_actions):
        """ Constructor from scratch """
        
        self.sensors = np.zeros(num_sensors)
        self.primitives = np.zeros(num_primitives)
        self.actions = np.zeros(num_actions)
        
    def zeros_like(self):
        """  Create a new state instance the same size as old_state, 
        but all zeros
        """
        
        zero_state = copy.deepcopy(self)
        zero_state.sensors = np.zeros_like(self.sensors)
        zero_state.primitives = np.zeros_like(self.primitives)
        zero_state.actions = np.zeros_like(self.actions)
        
        zero_state.features = [np.zeros_like(f) for f in self.features]

        return zero_state
        

    def integrate_state(self, new_state, decay_rate):
        """ Updates the state by combining the new state value with a 
        decayed version of the current state value.
        """
    
        integrated_state = State.zeros_like(self, new_state)
        integrated_state.sensors = utils.bounded_sum(
                                    self.sensors * (1 - decay_rate), 
                                    new_state.sensors)
        integrated_state.primitives = utils.bounded_sum(
                                    self.primitives * (1 - decay_rate), 
                                    new_state.primitives)
        integrated_state.actions = utils.bounded_sum(
                                    self.actions * (1 - decay_rate), 
                                    new_state.actions)

        for index in range(1, len(self.features)):
            integrated_state.features[index] = utils.bounded_sum(
                                    self.features[index] * (1 - decay_rate), 
                                    new_state.features[index])
                    
        return integrated_state
            
            
    def decay(self, factor):
        """ Decay all values of the state by a constant factor.
        Assumes factor is a scalar 0 <= factor < 1
        """
        self.sensors *= factor
        self.primitives *= factor
        self.actions *= factor

        for i in range(len(self.features)):
            self.features[i] *= factor 


    def bounded_sum(self, other_state):
        """ Add another State to this State, 
        ensuring that no value has a magnitude greater than one """
        new_state = other_state.zeros_like()
        new_state.sensors = utils.bounded_sum(self.sensors, 
                                              other_state.sensors)
        new_state.primitives = utils.bounded_sum(self.primitives, 
                                                 other_state.primitives)
        new_state.actions = utils.bounded_sum(self.actions, 
                                              other_state.actions)
        
        for i in range(len(self.features)):
            new_state.features[i] = utils.bounded_sum(self.features[i], 
                                                      other_state.features[i])

        return new_state

        
    def add_group(self):
        self.features.append(np.array([]))
        return None
        
    def add_feature(self, nth_group, value=0):
        self.features[nth_group] = np.hstack((self.features[nth_group], value))
        return None
         
    def n_feature_groups(self):
        return len(self.features)
    