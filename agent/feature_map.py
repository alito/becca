import numpy as np

class FeatureMap(object):

    def __init__(self, num_sensors, num_primitives, num_actions):
        self.sensors = np.zeros((num_sensors, num_sensors))
        self.primitives = np.eye(num_primitives)
        self.action = np.eye(num_actions)
        self.features = []


    def add_group(self, n_group_features, n_group_inputs):
        self.features.append(np.zeros((n_group_features, n_group_inputs)))
        
        for indx in range(n_group_features):
            feature = np.random.random_sample(n_group_inputs)
            feature = feature / np.linalg.norm(feature)
            self.features[-1][indx,:] = feature
            
