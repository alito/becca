import copy
import numpy as np
import utils

class State(object):
    """ A data structure for representing the internal state of the agent """ 

    def __init__(self, num_sensors=1, num_primitives=1, num_actions=1):
        """ Constructor from scratch """
        
        self.sensors = np.zeros((num_sensors,1))
        self.primitives = np.zeros((num_primitives,1))
        self.action = np.zeros((num_actions,1))
        self.features = []
        
        
    def zeros_like(self, dtype=np.float):
        """  Create a new state instance the same size as old_state, 
        but all zeros
        """
        state_type = dtype
        zero_state = copy.deepcopy(self)
        zero_state.sensors = np.zeros_like(self.sensors, dtype=state_type)
        zero_state.primitives = np.zeros_like(self.primitives, dtype=state_type)
        zero_state.action = np.zeros_like(self.action, dtype=state_type)
        
        zero_state.features = [np.zeros_like(f, dtype=state_type) 
                               for f in self.features]
        return zero_state
        

    def add_group(self, num_features, new_array=None, dtype=np.float):
        """ Add a group with a known number of features """
        group_type = dtype
        if new_array == None:
            self.features.append(np.zeros((num_features,1), dtype=group_type))
        else:
            self.features.append(new_array)
            
        return None
        
        
    def unbounded_sum(self, other_state):
        """ Add another State to this State. 
        Values of individual features may have a magnitude greater than 1.
        """
        new_state = other_state.zeros_like()
        
        new_state.sensors = self.sensors + other_state.sensors
        new_state.primitives = self.primitives + other_state.primitives
        new_state.action = self.action + other_state.action
        
        for i in range(self.n_feature_groups()):
            new_state.features[i] = self.features[i] + other_state.features[i]
            
        return new_state
        

    def bounded_sum(self, other_state):
        """ Add another State to this State, 
        ensuring that no value has a magnitude greater than one """
        new_state = other_state.zeros_like()
        
        new_state.sensors = utils.bounded_sum(self.sensors, 
                                              other_state.sensors)
        new_state.primitives = utils.bounded_sum(self.primitives, 
                                                 other_state.primitives)
        new_state.action = utils.bounded_sum(self.action, 
                                              other_state.action)
        
        for i in range(self.n_feature_groups()):
            new_state.features[i] = utils.bounded_sum(self.features[i], 
                                                      other_state.features[i])
        return new_state

    
    def multiply(self, multiplier):
        """ Multiply this State by a scalar """
        new_state = self.zeros_like()
        
        new_state.sensors = self.sensors * multiplier
        new_state.primitives = self.primitives * multiplier
        new_state.action = self.action * multiplier
        
        for i in range(self.n_feature_groups()):
            new_state.features[i] = self.features[i] * multiplier
            
        return new_state

    
    def integrate_state(self, new_state, decay_rate):
        """ Updates the state by combining the new state value with a 
        decayed version of the current state value.
        """
    
        integrated_state = new_state.zeros_like()
        integrated_state.sensors = utils.bounded_sum(
                                    self.sensors * (1 - decay_rate), 
                                    new_state.sensors)
        integrated_state.primitives = utils.bounded_sum(
                                    self.primitives * (1 - decay_rate), 
                                    new_state.primitives)
        integrated_state.action = utils.bounded_sum(
                                    self.action * (1 - decay_rate), 
                                    new_state.action)

        for index in range(self.n_feature_groups()):
            integrated_state.features[index] = utils.bounded_sum(
                                    self.features[index] * (1 - decay_rate), 
                                    new_state.features[index])
                    
        return integrated_state
            
            
    def n_feature_groups(self):
        return len(self.features)
    
       
    def n_features_in_group(self, group_index):
        return self.features[group_index].size
    