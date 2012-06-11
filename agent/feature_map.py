import numpy as np

class FeatureMap(object):

    def __init__(self, num_sensors, num_real_primitives, num_actions):
        self.sensors = np.zeros((num_sensors, num_sensors))
        self.primitives = np.eye(num_real_primitives)
        self.actions = np.eye(num_actions)
        self.features = []


    def add_group(self, n_group_inputs):
        self.features.append(np.zeros((0, n_group_inputs)))


    def add_feature(self, group, feature):
        """ Add a feature to group """
        self.features[group] = \
                np.vstack((self.features[group], feature))

