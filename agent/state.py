
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
        
    def zeros_like(self, old_state):
        """  Create a new state instance the same size as old_state, 
        but all zeros
        """
        
        self.sensors = np.zeros_like(old_state.sensors)
        self.primitives = np.zeros_like(old_state.primitives)
        self.actions = np.zeros_like(old_state.actions)
        
        self.features = [np.zeros_like(f) for f in old_state.features]
        

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
            
    def add_group(self):
        self.features.append(np.array([]))