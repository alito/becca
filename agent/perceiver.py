import copy
import itertools
import numpy as np
import state
import utils
import viz_utils

class Perceiver(object):
    """ The object responsible for feature extraction. 
    This includes assembling inputs of various types,
    determining their co-activity, creating features,
    and using the feature map at each time step
    to translate the input into feature activity.
    """
    
    def __init__(self, num_sensors, num_primitives, num_actions,
                 max_num_features):

        """ Control how rapidly previous inputs are forgotten """
        self.INPUT_DECAY_RATE = 1.0                 # real, 0 < x < 1
        
        """ Control how rapidly the co-activity update plasticity changes """
        self.PLASTICITY_UPDATE_RATE = 10. ** (-3)   # real, 0 < x < 1, small
        
        """ The exponent used in dividing inputs' energy between the 
        features that they activate.
        """
        self.ACTIVATION_WEIGHTING_EXPONENT = 10     # real, 1 < x 
        
        """ Determines how much an input's contribution to exciting features
        dissipates its contribution to the co-activity estimate.
        """
        self.DISSIPATION_FACTOR = 0.3               # real, 0 < x 
                
        """ Once a co-activity value exceeds this value, 
        nucleate a new feature.
        """ 
        self.NEW_FEATURE_THRESHOLD = 0.1            # real,  x >= 0
        
        """ If the minimum co-activity between each 
        of the elements of a growing feature and the next candidates 
        is lower than this value, don't add any more. 
        """
        self.MIN_SIG_COACTIVITY = 0.98  * self.NEW_FEATURE_THRESHOLD
        # real,  0 < x <= self.NEW_FEATURE_THRESHOLD

        """ Stop creating new groups, once this number of features 
        is nearly reached.
        """
        self.MAX_NUM_FEATURES = max_num_features
        
        """ A flag determining whether to stop creating new groups """
        self.features_full = False
        
        """ The list of 2D arrays that translates grouped inputs 
        into feature activities.
        """
        self.feature_map = np.zeros(
                        (self.MAX_NUM_FEATURES, 
                         self.MAX_NUM_FEATURES + num_sensors))
        
        """ 2D array for holding the estimate of the co-activity """
        self.coactivity = np.zeros(
                        (self.MAX_NUM_FEATURES + num_sensors, 
                         self.MAX_NUM_FEATURES + num_sensors))

        """ 2D array for tracking the allowable 
        combinations of elements.  
        """
        self.combination = np.ones(self.coactivity.shape) - \
                           np.eye(self.coactivity.shape[0])
                           
        """ Disallow combinations of actions with any other feature
        for forming features. It's unclear still whether this is 
        the best appraoch, but it works with the current benchmark suite. 
        """
        self.combination[num_sensors + num_primitives: 
                         num_sensors + num_primitives + num_actions,:] = 0
        
        """ State that provides memory of the input on 
        the previous time step. 
        """
        self.previous_input = state.State(num_primitives, num_actions, 
                                          self.MAX_NUM_FEATURES)
        
        """ The activity levels of all features in all groups.
        Passed on to the reinforcement learner at each time step.
        """
        self.feature_activity = self.previous_input.zeros_like()

        self.n_features = num_primitives
        self.n_features += num_actions
        self.n_sensors = num_sensors
        

    def step(self, sensors, primitives, actions):
        """ Incrementally estimate co-activity between inputs 
        and create new features when appropriate.
        """

        """ Make sure all the inputs are 2D arrays """
        if len(sensors.shape) == 1:
            sensors = sensors[:,np.newaxis]
        if len(primitives.shape) == 1:
            primitives = primitives[:,np.newaxis]
        if len(actions.shape) == 1:
            actions = actions[:,np.newaxis]
        
        """ Build the input vector.
        Combine sensors and primitives with 
        previous feature_activity to create the full input set.
        """        
        new_feature_input = copy.deepcopy(self.feature_activity)
        new_feature_input.set_primitives(primitives)

        """ It's not yet clear whether actions should be included or not """
        new_feature_input.set_actions(actions)
        #new_feature_input.set_actions(np.zeros(actions.shape))

        """ Decay previous input and combine it with the new input """
        self.previous_input.multiply(1 - self.INPUT_DECAY_RATE)
        new_feature_input = new_feature_input.bounded_sum(self.previous_input)
        new_input = new_feature_input.prepend(sensors)
        
        """ Update previous input, preparing it for the next iteration """    
        self.previous_input = copy.deepcopy(new_feature_input)
        
        """ Truncate to currently-used features """
        new_input = new_input[:self.n_sensors + self.n_features,:]

        coactivity_inputs = self.calculate_feature_activities(new_input)
        self.feature_activity.set_primitives(primitives)
        self.feature_activity.set_actions(actions)
         
        """ As appropriate, update the co-activity estimate and 
        create new features.
        """                 
        if not self.features_full:
            self.update_coactivity_matrix(coactivity_inputs)
            self.create_new_features()

        return self.feature_activity, self.n_features
        

    def calculate_feature_activities(self, new_input):
        """ Make a first pass at the feature activation levels by 
        multiplying across the feature map.
        """
        current_feature_activities = np.dot(self.feature_map \
                                [:self.n_features, :new_input.size], new_input)

        """ Find the activity levels of the features contributed to by each
        input.
        """
        feature_contribution_map = np.zeros((self.n_features, new_input.size))
        feature_contribution_map[np.nonzero(self.feature_map
                                [:self.n_features, :new_input.size])] = 1.
        activated_feature_map = current_feature_activities * \
                                            feature_contribution_map
        
        """ Find the largest feature activity that each input 
        contributes to.
        """
        max_activation = np.max(activated_feature_map, axis=0) + \
                                utils.EPSILON

        """ Divide the energy that each input contributes to each feature, 
        based on an exponential weighting. More active features take
        more of the energy. Less active features are further starved.
        The ACTIVATION_WEIGHTING_EXPONENT controls the extent to which this 
        happens. If it's infinite, then this degenerates to 
        winner-take-all. It's meant to mimic a competitive lateral 
        inhibition mechanism.
        """
        input_inhibition_map = np.power(activated_feature_map / max_activation, 
                                        self.ACTIVATION_WEIGHTING_EXPONENT)
        
        """ Find the effective strength of each input to each 
        feature after inhibition.
        """
        inhibited_inputs = input_inhibition_map * new_input.transpose()

        final_feature_activities = np.sum( self.feature_map \
               [:self.n_features, :new_input.size] * inhibited_inputs, axis=1)
        self.feature_activity.set_features(final_feature_activities
                                           [:self.n_features])

        """ Calculate how much energy each input has left to contribute
        to the co-activity estimate. Discounting each input according
        to how stongly it contributed to active features mimics a
        biological process in which each input has a limited amount of
        energy to spend. Combining with other inputs to create new features
        also requires energy. Handling energy in this way results in a 
        self-limiting process in which inputs contribute to a limited 
        number of features.
        """
        combined_weights = np.sum(input_inhibition_map, axis=0) + utils.EPSILON
        coactivity_inputs = new_input * \
                            2 ** (-combined_weights[:, np.newaxis] * 
                                  self.DISSIPATION_FACTOR)
        return coactivity_inputs


    def update_coactivity_matrix(self, new_input):
        """ Update an estimate of co-activity between every
        feature and every other feature, including the sensors, primitives,
        and action. 
        """
        instant_coactivity = np.dot(new_input, new_input.transpose())
        
        """ Determine the upper bound on the size of the incremental step 
        toward the instant co-factivity. It is weighted by the 
        feature associated with the column of the co-activity estimate. 
        Weighting by the feature column introduces an asymmetry, 
        that is, the co-activity of feature A with feature B 
        is not necessarily the same as the co-activity of
        feature B with feature A.
        """
        delta_coactivity = np.tile(new_input.transpose(), \
                     (new_input.size, 1)) * \
                     (instant_coactivity - \
                     self.coactivity[:new_input.size, :new_input.size])
                     
        """ Adapt co-activity toward instant co-activity by
        the calculated step size at the prescibed rate.
        """
        self.coactivity[:new_input.size, :new_input.size] += \
                     self.PLASTICITY_UPDATE_RATE * delta_coactivity
        return
    
    
    def create_new_features(self):
        """ If the right conditions have been reached,
        create a new feature.
        """    
        n_inputs = self.n_sensors + self.n_features
        mutual_coactivity = np.minimum(
                       self.coactivity[:n_inputs, :n_inputs], \
                       self.coactivity[:n_inputs, :n_inputs].transpose())
        
        """ Make sure that disallowed combinations are not used to 
        nucleate new features. They can, however, be agglomerated onto 
        those features after they are nucleated.
        """
        mutual_coactivity_nuclei = mutual_coactivity * \
                            self.combination[:n_inputs, :n_inputs] * \
                            self.combination[:n_inputs, :n_inputs].transpose()

        max_coactivity = np.max(mutual_coactivity_nuclei)
        
        if max_coactivity > self.NEW_FEATURE_THRESHOLD:
            
            """ Nucleate a new feature under the two elements for which 
            co-activity is a maximum.
            """
            indices1, indices2 = (mutual_coactivity_nuclei ==
                                  max_coactivity).nonzero()

            which_index = np.random.random_integers(0, len(indices1)-1)
            index1 = indices1[which_index]
            index2 = indices2[which_index]
            added_feature_indices = [index1, index2]
            
            #debug
            print 'new feature nucleated with indices ', index1, 'and', index2, \
                    'with a co-activity of', max_coactivity
            
            for element in itertools.product(added_feature_indices, 
                                             added_feature_indices):
                mutual_coactivity[element] = 0
            
            """ Add elements one at a time in a greedy fashion until 
            either the maximum number of elements is reached or 
            the minimum amount of co-activity with the other members 
            of the group cannot be achieved.
            """
            coactivity_threshold = self.MIN_SIG_COACTIVITY
            while True:
                candidate_match_strength = np.min(mutual_coactivity
                                                  [:,added_feature_indices],
                                                  axis=1)
                
                """ Find the next most co-active feature and test 
                whether its co-activity is high enough to add it to the
                group.
                """ 
                if (np.max(candidate_match_strength) <= coactivity_threshold):
                    break

                max_match_strength = np.max(candidate_match_strength)
                max_match_strength_indices = (candidate_match_strength == 
                                      max_match_strength).nonzero()[0]
                index = max_match_strength_indices[
                           np.random.random_integers(0, 
                           len(max_match_strength_indices)-1)]
                
                """ Add the new input to the feature """
                added_feature_indices.append(index)
                
                """ Update the co-activity estimate """
                for element in itertools.product(added_feature_indices, 
                                                 added_feature_indices):
                    mutual_coactivity[element] = 0

            """ Add the new feature to the feature map """
            self.feature_map[self.n_features, added_feature_indices] = \
                                1. / np.float(len(added_feature_indices))  
            
            self.n_features += 1
            self.disallow_generation_crossing(added_feature_indices)
            
            #debug
            print 'adding feature', self.n_features, 'in position', \
                    self.n_features + self.n_sensors - 1, 'with inputs', \
                        added_feature_indices
                       
            """ Check to see whether the capacity to store and update features
            has been reached.
            """
            if self.n_features >= self.MAX_NUM_FEATURES:
                self.features_full = True
                print('==Max number of features reached (' + \
                      str(self.MAX_NUM_FEATURES) + ')==') 
        return 

          
    def disallow_generation_crossing(self, added_feature_indices):
        """ Find the elements that cannot be grouped with the parents of
        the feature to form new groups and explicitly prohibit them
        from doing so. The co-activity between these is forced 
        to be zero to discourage these being combined 
        to create new features.
        """
        
        new_index = self.n_features + self.n_sensors - 1
        
        """ Adopt the disallowed combinations of all parents """
        for element in added_feature_indices:
            self.combination[new_index,:] = np.minimum( 
                self.combination[new_index,:], self.combination[element,:])
            
        """ Disallow combinations with any of the parents too """
        self.combination[new_index,added_feature_indices] = 0
            
        """ Disallow combinations between the parents """
        for element in itertools.product(added_feature_indices, 
                             added_feature_indices):
            self.combination[element] = 0
        
        return 
        
        
    def visualize(self, save_eps=False):
        n_inputs = self.n_sensors + self.n_features
        
        mutual_coactivity = np.minimum(self.coactivity[:n_inputs, :n_inputs], \
                    self.coactivity[:n_inputs, :n_inputs].transpose())

        viz_utils.visualize_coactivity(mutual_coactivity, n_inputs, save_eps)
        viz_utils.visualize_feature_map(self.feature_map[:self.n_features, 
                                                         :n_inputs])
        viz_utils.force_redraw()
        return
    