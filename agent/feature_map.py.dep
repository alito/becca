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
            
            
    def get_receptive_field(self, group_index, feature_index):
        num_features = self.features[group_index].shape[0]
        feature = self.features[group_index][feature_index,:]
        
        """ Find the centroid of all the features """
        centroid = np.zeros(feature.shape)
        for feature_counter in range(num_features):
            centroid += self.features[group_index][feature_counter,:] / \
                            num_features
                            
        """ Find the rotation direction """ 
        rotation_direction = feature - centroid
        
        """ Try to rotate the feature vector as far as it will go, 
        i.e. until one component becomes zero. At least one component of 
        the rotation direction vector should be negative.
        """
        rotation_amount_candidates = -feature / (rotation_direction + 10 ** -6)
        
        """ Only consider positive amounts. These are away from the 
        centroid, rather than toward it.
        """
        rotation_amount_candidates = \
                    np.extract(rotation_amount_candidates >= 0, 
                               rotation_amount_candidates)
        
        if rotation_amount_candidates.size == 0:
            return feature
        else:
            rotation_amount = np.min(rotation_amount_candidates)
            receptive_field_unscaled = feature + rotation_direction * rotation_amount
            receptive_field = receptive_field_unscaled / \
                            np.max(receptive_field_unscaled)
                            
            return(receptive_field)