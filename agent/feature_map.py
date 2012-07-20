import numpy as np

class FeatureMap(object):

    def __init__(self, num_sensors, num_primitives, num_actions):
        self.sensors = np.zeros((num_sensors, num_sensors))
        self.primitives = np.eye(num_primitives)
        self.action = np.eye(num_actions)
        self.features = []


    def add_fixed_group(self, n_group_features, n_group_inputs):
        self.features.append(np.zeros((n_group_features, n_group_inputs)))
        
        for indx in range(n_group_features):
            feature = np.random.random_sample(n_group_inputs)
            feature = feature / np.linalg.norm(feature)
            self.features[-1][indx,:] = feature
            
        '''def add_group(self, n_group_inputs):
        self.features.append(np.zeros((0, n_group_inputs)))


        def add_feature(self, group, feature):
        """ Add a feature to group """
        self.features[group] = \
                np.vstack((self.features[group], feature))
        '''

    def size(self):
        """ Determine the approximate number of elements being used by the
        class and its members. Created to debug an apparently excessive 
        use of memory.
        """
        total = 0
        total += self.sensors.size
        total += self.primitives.size
        total += self.action.size
        for group_index in range(len(self.features)):
            total += self.features[group_index].size

        return total
            
            

