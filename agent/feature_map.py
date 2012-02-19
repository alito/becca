import numpy as np

class FeatureMap(object):

    def __init__(self, num_sensors, num_primitives, num_actions):
        self.NEW_FEATURE_MARGIN = 0.3
        self.NEW_FEATURE_MIN_SIZE = 0.2

        self.map = []
        self.map.append(np.eye(num_sensors))
        self.map.append(np.eye(num_primitives))
        self.map.append(np.eye(num_actions))


    def add_group(self, group_length):
        self.map.append(np.zeros((group_length)))


    def add_feature(self, group, has_dummy, feature):
        """
        Add a feature to group
        """
        self.map[group] = np.vstack((self.map[group], np.transpose(feature)))

        
