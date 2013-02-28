
from utils import *
import viz_utils

import itertools
import numpy as np

class Perceiver(object):
    """ The object responsible for feature extraction. 
    This includes assembling inputs of various types, determining their co-activity, creating features,
    and using the feature map at each time step to translate the input into feature activity.
    """
    
    def __init__(self, num_sensors, num_primitives, num_actions,
                 max_num_features):

        self.num_raw_sensors = num_sensors
        self.num_sensors = num_sensors
        self.num_primitives = num_primitives
        self.num_actions = num_actions
        self.num_features = num_primitives + num_actions

        """ Once a co-activity value exceeds this value, nucleate a new feature """ 
        self.NEW_FEATURE_THRESHOLD = 0.1            # real,  x >= 0
        
        """ If the minimum co-activity between each of the elements of a growing feature 
        and the next candidates is lower than this value, don't add any more. 
        """
        self.MIN_SIG_COACTIVITY =  0.7 * self.NEW_FEATURE_THRESHOLD # real,  0 < x <= 1.0
        self.PLASTICITY_UPDATE_RATE = 0.01 * self.NEW_FEATURE_THRESHOLD # real, 0 < x < 1, small
        
        """ Determines how much an input's contribution to exciting features
        dissipates its contribution to the co-activity estimate.
        """
        self.DISSIPATION_FACTOR = 0.3               # real, 0 < x 
                
        """ The exponent used in dividing inputs' energy between the features that they activate """
        self.ACTIVATION_WEIGHTING_EXPONENT = 10     # real, 1 < x 
                
        self.max_num_features = max_num_features
        self.features_full = False        
        self.previous_input = np.zeros((self.max_num_features, 1))
        self.feature_activity = np.zeros((self.max_num_features, 1))
        self.feature_map = np.zeros((self.max_num_features, self.max_num_features + self.num_sensors))
        self.coactivity = np.zeros((self.max_num_features + self.num_sensors, 
                                    self.max_num_features + self.num_sensors))
        self.combination = np.ones(self.coactivity.shape) - np.eye(self.coactivity.shape[0])
                           
        """ Disallow combinations of actions with any other feature for forming features. 
        It's unclear still whether this is the best appraoch, but it works with the current benchmark suite. 
        """
        self.combination[self.num_sensors + self.num_primitives: 
                         self.num_sensors + self.num_primitives + self.num_actions,:] = 0
                
        self.sensor_min = np.ones((self.num_raw_sensors, 1)) * BIG
        self.sensor_max = np.ones((self.num_raw_sensors, 1)) * (-BIG)
        self.SENSOR_RANGE_DECAY_RATE = 10 ** -3


    def step(self, raw_sensors, primitives, actions):

        """ Make sure all the inputs are 2D arrays """
        if len(raw_sensors.shape) == 1:
            raw_sensors = raw_sensors[:,np.newaxis]
        if len(primitives.shape) == 1:
            primitives = primitives[:,np.newaxis]
        if len(actions.shape) == 1:
            actions = actions[:,np.newaxis]
        
        """ Modify sensor inputs so that they fall between 0 and 1 """
        self.sensor_min = np.minimum(raw_sensors , self.sensor_min)
        self.sensor_max = np.maximum(raw_sensors , self.sensor_max)
        spread = self.sensor_max - self.sensor_min
        sensors = (raw_sensors - self.sensor_min) / (spread + EPSILON)
        self.sensor_min += spread * self.SENSOR_RANGE_DECAY_RATE
        self.sensor_max -= spread * self.SENSOR_RANGE_DECAY_RATE
        
        """ Combine sensors and primitives with previous feature_activity to create the full input set """        
        new_feature_input = np.copy(self.feature_activity)
        new_feature_input[:self.num_primitives,:] = primitives
        new_feature_input[self.num_primitives: self.num_primitives + self.num_actions,:] = actions    
        new_input = np.vstack((sensors, new_feature_input))        
        self.previous_input = np.copy(new_feature_input)
        
        """ Truncate to currently-used features """
        new_input = new_input[:self.num_sensors + self.num_features,:]
        
        coactivity_inputs = self.calculate_feature_activities(new_input)
        self.feature_activity[:self.num_primitives,:] = new_feature_input[:self.num_primitives,:]
        self.feature_activity[self.num_primitives: self.num_primitives + self.num_actions,:] = \
            new_feature_input[self.num_primitives: self.num_primitives + self.num_actions,:]
         
        """ As appropriate, update the co-activity estimate and create new features """                 
        if not self.features_full:
            self.update_coactivity_matrix(coactivity_inputs)
            self.create_new_features()

        return self.feature_activity, self.num_features
        

    def calculate_feature_activities(self, new_input):
        """ Make a first pass at the feature activation levels by multiplying across the feature map """
        initial_feature_activities = np.dot(self.feature_map [:self.num_features, :new_input.size], new_input)

        """ Find the activity levels of the features contributed to by each input """
        feature_contribution_map = np.zeros((self.num_features, new_input.size))
        feature_contribution_map[np.nonzero(self.feature_map[:self.num_features, :new_input.size])] = 1.
        activated_feature_map = initial_feature_activities * feature_contribution_map
        
        """ Find the largest feature activity that each input contributes to """
        max_activation = np.max(activated_feature_map, axis=0) + EPSILON

        """ Divide the energy that each input contributes to each feature """
        input_inhibition_map = np.power(activated_feature_map / max_activation, 
                                        self.ACTIVATION_WEIGHTING_EXPONENT)
        
        """ Find the effective strength of each input to each feature after inhibition """
        inhibited_inputs = input_inhibition_map * new_input.transpose()
        final_feature_activities = np.sum( self.feature_map \
               [:self.num_features, :new_input.size] * inhibited_inputs, axis=1)
        self.feature_activity[:self.num_features,0] = final_feature_activities[:self.num_features]

        """ Calculate how much energy each input has left to contribute to the co-activity estimate """
        final_activated_feature_map = final_feature_activities[:,np.newaxis] * feature_contribution_map
        combined_weights = np.sum(final_activated_feature_map, axis=0) + EPSILON
        coactivity_inputs = new_input * 2 ** (-combined_weights[:, np.newaxis] * self.DISSIPATION_FACTOR)
        return coactivity_inputs


    def update_coactivity_matrix(self, new_input):
        """ Update an estimate of co-activity between every feature and every other feature """
        instant_coactivity = np.dot(new_input, new_input.transpose())
        
        """ Determine the upper bound on the size of the incremental step toward the instant co-activity """
        delta_coactivity = np.tile(new_input.transpose(), (new_input.size, 1)) * \
                     (instant_coactivity - self.coactivity[:new_input.size, :new_input.size])
                     
        """ Adapt co-activity toward instant co-activity by the calculated step size at the prescibed rate """
        self.coactivity[:new_input.size, :new_input.size] += self.PLASTICITY_UPDATE_RATE * delta_coactivity
        return
    
    
    def create_new_features(self):
        """ If the right conditions have been reached, create a new feature """    
        n_inputs = self.num_sensors + self.num_features
        mutual_coactivity = np.minimum(self.coactivity[:n_inputs, :n_inputs], \
                                       self.coactivity[:n_inputs, :n_inputs].transpose())
        
        """ Make sure that disallowed combinations are not used to nucleate new features """
        mutual_coactivity_nuclei = mutual_coactivity * self.combination[:n_inputs, :n_inputs] * \
                                                       self.combination[:n_inputs, :n_inputs].transpose()
        max_coactivity = np.max(mutual_coactivity_nuclei)
        if max_coactivity > self.NEW_FEATURE_THRESHOLD:
            
            """ Nucleate a new feature under the two elements for which co-activity is a maximum """
            indices1, indices2 = (mutual_coactivity_nuclei == max_coactivity).nonzero()
            which_index = np.random.random_integers(0, len(indices1)-1)
            index1 = indices1[which_index]
            index2 = indices2[which_index]
            added_feature_indices = [index1, index2]
            for element in itertools.product(added_feature_indices, added_feature_indices):
                mutual_coactivity[element] = 0
            
            """ Add elements one at a time in a greedy fashion """
            coactivity_threshold = self.MIN_SIG_COACTIVITY
            while True:
                candidate_match_strength = np.min(mutual_coactivity[:,added_feature_indices], axis=1)
                
                """ Find the next most co-active feature """ 
                if (np.max(candidate_match_strength) <= coactivity_threshold):
                    break
                max_match_strength = np.max(candidate_match_strength)
                max_match_strength_indices = (candidate_match_strength == max_match_strength).nonzero()[0]
                index = max_match_strength_indices[np.random.random_integers(0, 
                                                    len(max_match_strength_indices)-1)]
                added_feature_indices.append(index)
                for element in itertools.product(added_feature_indices, added_feature_indices):
                    mutual_coactivity[element] = 0

            """ Add the new feature to the feature map """
            self.feature_map[self.num_features, added_feature_indices] = \
                                1. / np.float(len(added_feature_indices))  
            self.num_features += 1
            self.disallow_generation_crossing(added_feature_indices)
            
            #debug
            #print 'adding feature', self.num_features, 'in position', \
            #        self.num_features + self.num_sensors - 1, 'with inputs', added_feature_indices
            
            if self.num_features >= self.max_num_features:
                self.features_full = True
                print('==Max number of features reached (' + str(self.max_num_features) + ')==') 
        return 

          
    def disallow_generation_crossing(self, added_feature_indices):
        """ Find the elements that cannot be grouped with the parents of the feature """
        new_index = self.num_features + self.num_sensors - 1
        
        """ Adopt the disallowed combinations of all parents """
        for element in added_feature_indices:
            self.combination[new_index,:] = np.minimum( 
                self.combination[new_index,:], self.combination[element,:])
            
        """ Disallow combinations with any of the parents too """
        self.combination[new_index,added_feature_indices] = 0
            
        """ Disallow combinations between the parents """
        for element in itertools.product(added_feature_indices, added_feature_indices):
            self.combination[element] = 0
        return 
        
        
    def visualize(self, save_eps=False):
        n_inputs = self.num_sensors + self.num_features
        mutual_coactivity = np.minimum(self.coactivity[:n_inputs, :n_inputs], \
                    self.coactivity[:n_inputs, :n_inputs].transpose())
        viz_utils.visualize_coactivity(mutual_coactivity, n_inputs, save_eps)
        viz_utils.visualize_feature_map(self.feature_map[:self.num_features, :n_inputs])
        viz_utils.force_redraw()
        return
    