import numpy as np

class FeatureMap(object):

    def __init__(self, num_sensors, num_primitives, num_actions):
        self.map = []
        self.map.append(np.eye(num_sensors))
        self.map.append(np.eye(num_primitives))
        self.map.append(np.eye(num_actions))


    def add_group(self, group_length):
        self.map.append(np.zeros((1, group_length)))


    def add_feature(self, group, has_dummy, feature):
        """
        Add a feature to group
        """

        self.map[group] = np.vstack((self.map[group], feature)) 
        if has_dummy:
            self.map[group] = self.map[group][1:, :]

