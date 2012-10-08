import copy
import itertools
import numpy as np
import state
import viz_utils

class Perceiver(object):
    """ The object responsible for feature creation. 
    This includes assembling inputs of various types,
    determining how to group them, grouping them at each time step,
    building and updating the feature map, and using it at each time step
    to translate the input into feature activity.
    """
    
    def __init__(self, num_sensors, num_primitives, num_actions,
                 max_num_features):

        """ Control how rapidly previous inputs are forgotten """
        self.INPUT_DECAY_RATE = 1.0                 # real, 0 < x < 1
        
        """ Control how rapidly the coactivity update plasticity changes """
        self.PLASTICITY_UPDATE_RATE = 10. ** (-4)   # real, 0 < x < 1, small
        
        """ The maximum value of plasticity """
        #self.MAX_PLASTICITY = 0.3                   # real, 0 < x < 1
        
        """ The exponent used in dividing inputs' energy between the 
        features that they activate.
        """
        self.ACTIVATION_WEIGHTING_EXPONENT = 12      # real, 1 < x 
                
        """ The feature actvity penalty associated with 
        prior membership in other groups. 
        """
        #self.GROUP_DISCOUNT = 0.5                   # real, 0 < x < 1
        
        """ Once a coactivity value exceeds this value, 
        nucleate a new group.
        """ 
        self.NEW_FEATURE_THRESHOLD = 0.04              # real,  x >= 0
        
        """ If the coactivity between the first two group members 
        and the next candidates 
        is lower than this value, don't add it. 
        This value approaches NEW_FEATURE_THRESHOLD as the group size grows.
        See create_new_feature() below.
        """
        self.MIN_SIG_COACTIVITY = 0.03  # real,  x >= self.NEW_FEATURE_THRESHOLD

        """ The rate that threshold coactivity for adding new
        inputs to the group decays as new inputs are added.
        """
        #self.COACTIVITY_THRESHOLD_DECAY_RATE = 0.12 # real, 0 <= x < 1
        
        """ The number of features to be added at the creation of every new
        feature group as a fraction of the group's total number of inputs. 
        """
        #self.N_GROUP_FEATURES_FRACTION = 0.33
        
        """ Stop creating new groups, once this number of features 
        is nearly reached.
        """
        self.MAX_NUM_FEATURES = max_num_features
        
        """ Factor controlling the strength of the inhibitory effect 
        that neighboring features have on each other when excersizing
        mutual inhibition.
        """
        #self.INHIBITION_STRENGTH_FACTOR = 1.
        
        """ The rate at which features adapt toward observed input data """
        #self.FEATURE_ADAPTATION_RATE = 10. ** -1   # real, 0 <= x, small

        """ The rate at which feature fatigue decays.
        0 means it never decays and 1 means it decays immediately--that
        there is no fatigue.
        """
        #self.FATIGUE_DECAY_RATE = 10. ** -2        # real, 0 <= x, small
        
        """ The strength of the influence that fatigue has on the
        features.
        """
        #self.FATIGUE_SUSCEPTIBILITY = 10. ** -1    # real, 0 <= x < 1
        
        """ To prevent normalization from giving a divide by zero """
        self.EPSILON = 1e-6
        
        """ A flag determining whether to stop creating new groups """
        self.features_full = False
        
        """ The list of 2D arrays that translates grouped inputs 
        into feature activities.
        """
        #self.feature_map = FeatureMap(num_sensors, num_primitives, 
        #                              num_actions)        
        self.feature_map = np.zeros(
                        (self.MAX_NUM_FEATURES, 
                         self.MAX_NUM_FEATURES + num_sensors))
        
        """ 2D array for holding the estimate of the coactivity """
        self.coactivity = np.zeros(
                        (self.MAX_NUM_FEATURES + num_sensors, 
                         self.MAX_NUM_FEATURES + num_sensors))

        """ 2D array for tracking the allowable 
        combinations of elements.  
        """
        self.combination = np.ones(self.coactivity.shape) - \
                           np.eye(self.coactivity.shape[0])
        
        """ 2D array for tracking disallowed combinations. 
        A list of 1D numpy arrays, one for each feature, 
        giving the coactivity
        indices that features in that group are descended from. The coactivity
        between these is forced to be zero to discourage these being combined 
        to create new features.
        """
        #self.disallowed = []
        
        """ 2D array for tracking the propensity for individual
        coactivity values to change, i.e. their plasticty
        """
        # self.plasticity = np.zeros(self.coactivity.shape)
        
        """ 1D array recording the number of groups that each input
        feature is a member of
        """
        #self.groups_per_feature = np.zeros((self.MAX_NUM_FEATURES, 1))
        
        """ 1D array containing all the inputs from all the groups.
        Used in order to make updating the coactivity
        less computationally expensive. 
        """
        #self.input_activity = np.zeros((self.MAX_NUM_FEATURES, 1))
        
        """ State that provides memory of the input on 
        the previous time step 
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
        """ Incrementally estimate coactivity between inputs 
        and form groups when appropriate
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

        """ It's not yet clear whether this should be included or not """
        #new_feature_input.set_actions(actions)
        new_feature_input.set_actions(np.zeros(actions.shape))

        """ Decay previous input and combine it with the new input """
        self.previous_input = self.previous_input.multiply(1 - 
                                                self.INPUT_DECAY_RATE)
        new_feature_input = new_feature_input.bounded_sum(self.previous_input)
        
        """ Update previous input, preparing it for the next iteration """    
        self.previous_input = copy.deepcopy(new_feature_input)
        
        new_input = new_feature_input.prepend(sensors)

        coactivity_inputs = self.calculate_feature_activities(new_input)
        
        """ As appropriate, update the coactivity estimate and 
        create new groups.
        """                 
        if not self.features_full:
            self.update_coactivity_matrix(coactivity_inputs)
            self.create_new_feature()

        return self.feature_activity


    def calculate_feature_activities(self, new_input):
        """ Figure out what the feature activities are """
        """ Make a first pass at the feature activation levels by 
        multiplying across the feature map.
        """
        verbose = False
        if np.random.random_sample() < 10**(-7):
            verbose = True
            
        if verbose:
            print 'new_input'
            print new_input
            
        if verbose:
            print 'feature_map'
            print self.feature_map [:self.n_features, :new_input.size]
            
        current_feature_activities = np.dot(self.feature_map \
                                [:self.n_features, :new_input.size], new_input)
        if verbose:
            print 'current_feature_activities'
            print current_feature_activities

        """ Find the activity levels of the features contributed to by each
        input.
        """
        feature_contribution_map = np.zeros((self.n_features, new_input.size))
        feature_contribution_map[np.nonzero(self.feature_map
                                [:self.n_features, :new_input.size])] = 1.
        if verbose:
            print 'feature_contribution_map'
            print feature_contribution_map
        
        activated_feature_map = feature_contribution_map * \
                    np.tile(current_feature_activities, (1, new_input.size))
        if verbose:
            print 'activated_feature_map'
            print activated_feature_map
        
        combined_activations = np.sum(activated_feature_map, axis=0) + \
                                self.EPSILON
        if verbose:
            print 'combined_activations'
            print combined_activations
        
        """ Divide the energy that each input contributes to each feature, 
        based on an exponential weighting. More active features take
        more of the energy. Less active features are further starved.
        The ACTIVATION_WEIGHTING_EXPONENT controls the extent to which this 
        happens. If it's infinite, then this degenerates to 
        winner-take-all.
        """
        weighted_feature_map = activated_feature_map ** \
                                self.ACTIVATION_WEIGHTING_EXPONENT
        if verbose:
            print 'weighted_feature_map'
            print weighted_feature_map
        
        combined_weights = np.sum(weighted_feature_map, axis=0) + self.EPSILON
        if verbose:
            print 'combined_weights'
            print combined_weights
        
        energy_feature_map = weighted_feature_map / np.tile(combined_weights, 
                                                        (self.n_features, 1))
        if verbose:
            print 'energy_feature_map'
            print energy_feature_map
        
        split_inputs = energy_feature_map * new_input.transpose()
        
        if verbose:
            print 'split_inputs'
            print split_inputs
            
        weighted_feature_activities = np.sum( self.feature_map \
                   [:self.n_features, :new_input.size] * split_inputs, axis=1)
        if verbose:
            print 'weighted_feature_activities'
            print weighted_feature_activities

        coactivity_inputs = new_input
        #coactivity_inputs = new_input * \
        #                    2 ** (-combined_activations[:, np.newaxis])
                                 
        #self.feature_activity.set_features(weighted_feature_activities
        #                                   [:self.n_features])
        self.feature_activity.set_features(coactivity_inputs[:self.n_features])
        
        if verbose:
            print 'coactivity_inputs'
            print coactivity_inputs
            
        if verbose:
            show_feature = self.n_features - 1
            print 'show_feature'
            print show_feature
            print 'new_input'
            print new_input [show_feature, :]
            print 'feature_map'
            print self.feature_map [show_feature, :new_input.size]
            print 'current_feature_activities'
            print current_feature_activities[show_feature, :]
            print 'feature_contribution_map'
            print feature_contribution_map[show_feature, :new_input.size]
            print 'activated_feature_map'
            print activated_feature_map[show_feature, :new_input.size]
            print 'combined_activations'
            print combined_activations[show_feature]
            print 'weighted_feature_map'
            print weighted_feature_map[show_feature, :new_input.size]
            print 'combined_weights'
            print combined_weights[show_feature]
            print 'energy_feature_map'
            print energy_feature_map[show_feature, :new_input.size]
            print 'split_inputs'
            print split_inputs[show_feature, :new_input.size]
            print 'weighted_feature_activities'
            print weighted_feature_activities[show_feature]
            print 'coactivity_inputs'
            print coactivity_inputs[show_feature, :]
            

        return coactivity_inputs


    def update_coactivity_matrix(self, new_input):
        """ Update an estimate of coactivity between every
        feature and every other feature, including the sensors, primitives,
        and action. 
        """

        """ Find the upper bound on plasticity based on how many groups
        each feature is associated with.
        Then update the plasticity of each input to form new associations, 
        incrementally stepping each combintation's plasticity toward 
        its upper bound.
        """
        #self.plasticity[:new_input.size, :new_input.size] += \
        #    self.PLASTICITY_UPDATE_RATE * \
        #    (self.MAX_PLASTICITY - \
        #     self.plasticity[:new_input.size, :new_input.size])

        """ Decrease the magnitude of features if they are already 
        inputs to feature groups. The penalty is a negative exponential 
        in the number of groups that each feature belongs to. 
        """ 
        '''weighted_feature_vector = \
            (np.exp( - self.groups_per_feature [:self.n_features] * \
                     self.GROUP_DISCOUNT ) * \
                     self.input_activity[:self.n_features])
        '''
        """ Determine the coactivity value according to the 
        only the current inputs. It is the product of every weighted 
        feature activity with every other weighted feature activity.
        """
        #instant_coactivity = np.dot(weighted_feature_vector, 
        #             weighted_feature_vector.transpose())
        instant_coactivity = np.dot(new_input, new_input.transpose())
        
        """ Determine the upper bound on the size of the incremental step 
        toward the instant coactivity. It is weighted both by the 
        feature associated with the column of the coactivity estimate and
        the plasticity of each pair of elements. Weighting by the feature
        column introduces an asymmetry, that is the coactivity of feature
        A with feature B is not necessarily the same as the coactivity of
        feature B with feature A.
        """
        #delta_coactivity = np.tile(weighted_feature_vector.transpose(), \
        #             (self.n_features, 1)) * \
        #            (instant_coactivity - \
        #            self.coactivity[:self.n_features, :self.n_features])
        delta_coactivity = np.tile(new_input.transpose(), \
                     (new_input.size, 1)) * \
                     (instant_coactivity - \
                     self.coactivity[:new_input.size, :new_input.size])
                     
        #debug
        #if np.max(self.plasticity[:new_input.size, :new_input.size]*delta_coactivity) > 0.1:
        #    for row in range(delta_coactivity.shape[0]):
        #        print 'instant_coactivity, row', row
        #        print instant_coactivity[row,:]
        

        """ Adapt coactivity toward average activity coactivity by
        the calculated step size.
        """
        #self.coactivity[:new_input.size, :new_input.size] += \
        #             self.plasticity[:new_input.size, :new_input.size]* \
        #             delta_coactivity
        self.coactivity[:new_input.size, :new_input.size] += \
                     self.PLASTICITY_UPDATE_RATE * delta_coactivity
                     

        """ Update legal combinations in the coactivity matrix """
        #self.coactivity[:new_input.size, :new_input.size] *= \
        #    self.combination[:new_input.size, :new_input.size]

        """ Update the plasticity by subtracting the magnitude of the 
        coactivity change. 
        """
        #self.plasticity[:new_input.size, :new_input.size] = \
        #    np.maximum(self.plasticity[:new_input.size, 
        #                               :new_input.size] - \
        #    np.abs(delta_coactivity) * self.plasticity[:new_input.size, 
        #                               :new_input.size], 0)

        return
    
    
    def create_new_feature(self):
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
        mutual_coactivity_nuclei = mutual_coactivity * self.combination[:n_inputs, :n_inputs] * \
                            self.combination[:n_inputs, :n_inputs].transpose()

        max_coactivity = np.max(mutual_coactivity_nuclei)
        
        if max_coactivity > self.NEW_FEATURE_THRESHOLD:
            
            """ Nucleate a new group under the two elements for which 
            coactivity is a maximum.
            """
            indices1, indices2 = (mutual_coactivity_nuclei ==
                                  max_coactivity).nonzero()

            which_index = np.random.random_integers(0, len(indices1)-1)
            index1 = indices1[which_index]
            index2 = indices2[which_index]
            added_feature_indices = [index1, index2]
            
            print 'new feature nucleated with indices ', index1, 'and', index2, \
                    'with a coactivity of', max_coactivity
            
            for element in itertools.product(added_feature_indices, 
                                             added_feature_indices):
                mutual_coactivity[element] = 0

            """ Track the available elements with candidate_matches
            to add and the coactivity associated with each.
            """
            #candidate_matches = np.zeros((n_inputs, n_inputs))
            #candidate_matches[:,added_feature_indices] = 1.
            #candidate_matches[added_feature_indices,:] = 1.
            
            """ Add elements one at a time in a greedy fashion until 
            either the maximum 
            number of elements is reached or the minimum amount of
            coactivity with the other members of the group cannot be 
            achieved.
            """
            coactivity_threshold = self.MIN_SIG_COACTIVITY
            while True:
                """ Calculate overall coactivity of candidate features
                with all the elements already in the group using a combination 
                of column-wise and row-wise coactivity values. In true
                coactivity, these would be symmetric. In this 
                coactivity, they are not. 
                """
                #candidate_coactivities = np.abs(mutual_coactivity *
                #                                candidate_matches)
                #candidate_match_strength = np.sum(
                #            candidate_coactivities, axis=0) / \
                #            len(added_feature_indices)
                #candidate_match_strength = np.min(candidate_coactivities, 
                #                                  axis=0)
                candidate_match_strength = np.min(mutual_coactivity
                                                  [:,added_feature_indices],
                                                  axis=1)
                
                # debug
                #print mutual_coactivity[:,added_feature_indices]
                #print candidate_match_strength.ravel()
                
                """ Find the next most correlated feature and test 
                whether its coactivity is high enough to add it to the
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
                
                """ Add the new feature """
                added_feature_indices.append(index)
                
                """ Update the 2D arrays that estimate coactivity, 
                allowable combinations, and additional candidates for
                the group. """
                for element in itertools.product(added_feature_indices, 
                                                 added_feature_indices):
                    mutual_coactivity[element] = 0

                #candidate_matches[:,added_feature_indices] = 1.
                #candidate_matches[added_feature_indices,:] = 1.
                
                """ Update the coactivity threshold. This helps keep
                the group from becoming too large. This formulation
                results in an exponential decay 
                """
                #coactivity_threshold = coactivity_threshold + \
                #            self.COACTIVITY_THRESHOLD_DECAY_RATE * \
                #            (self.NEW_FEATURE_THRESHOLD - coactivity_threshold)

            self.feature_map[self.n_features, added_feature_indices] = \
                                1. / np.float(len(added_feature_indices))  
            
            self.n_features += 1
            self.disallow_generation_crossing(added_feature_indices)
            
            print 'adding feature', self.n_features, 'in position', \
                    self.n_features + self.n_sensors - 1, 'with inputs', \
                        added_feature_indices
                        
            #print self.combination[self.n_sensors + self.n_features - 1, \
            #                       :self.n_features + self.n_sensors - 1]

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
        from doing so.
        The coactivity
        between these is forced to be zero to discourage these being combined 
        to create new features.
        """
        
        new_index = self.n_features + self.n_sensors - 1
        for element in added_feature_indices:
            # adopt the disallowed combinations of all parents
            self.combination[new_index,:] = np.minimum( 
                self.combination[new_index,:], self.combination[element,:])
            
        # disallow combinations with any of the parents too
        self.combination[new_index,added_feature_indices] = 0
            
        # disallow combinations between the parents
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
    