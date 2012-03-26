import copy
import itertools
import numpy as np
import state

class Grouper(object):
    """ The object responsible for assembling inputs of various types,
    determining how to group them, and grouping them at each time step.
    Perform clustering on feature_vector to create feature groups. 
    """
    
    def __init__(self, num_sensors, num_actions, num_primitives, 
                 max_num_features):

        """ Control how rapidly previous inputs are forgotten """
        self.INPUT_DECAY_RATE = 1.0 # real, 0 < x < 1
        
        """ Control how rapidly the correlation update platicity changes """
        self.PLASTICITY_UPDATE_RATE = 10 ** (-3) # real, 0 < x < 1
        
        """ The maximum value of platicity """
        self.MAX_PROPENSITY = 0.1
        
        """ The feature actvity penalty associated with 
        prior membership in other groups. 
        """
        self.GROUP_DISCOUNT = 0.5
        
        """ Once a correlation value exceeds this value, 
        nucleate a new group.
        """ 
        self.NEW_GROUP_THRESHOLD = 0.3     # real,  x >= 0
        
        """ Once correlation between group members and new candidates 
        fall below this value, stop adding them.
        """
        self.MIN_SIG_CORR = 0.05  # real,  x >= 0
        
        """ Stop growing a group, once it reaches this size """
        self.MAX_GROUP_SIZE = 100
        
        """ Stop creating new groups, once this number of features 
        is nearly reached.
        """
        self.MAX_NUM_FEATURES = max_num_features

        """ A flag determining whether to stop creating new groups """
        self.features_full = False
        
        """ 2D array for holding the estimate of the pseudo-correlation """
        self.correlation = np.zeros(
                        (self.MAX_NUM_FEATURES, self.MAX_NUM_FEATURES))

        """ 2D array for tracking the allowable 
        combinations of elements 
        """
        self.combination = np.ones(
                        (self.MAX_NUM_FEATURES, self.MAX_NUM_FEATURES)) - \
                        np.eye( self.MAX_NUM_FEATURES)
        
        """ 2D array for tracking the propensity for individual
        pseudo-correlation values to change, i.e. their plasticty
        """
        self.platicity = np.zeros(
                        (self.MAX_NUM_FEATURES, self.MAX_NUM_FEATURES))
        
        """ 1D array recording the number of groups that each input
        feature is a member of
        """
        self.groups_per_feature = np.zeros(self.MAX_NUM_FEATURES)
        
        """ 1D array containing all the inputs from all the groups.
        Used in order to make updating the pseudo-correlation
        less computationally expensive. 
        """
        self.feature_vector = np.zeros(self.MAX_NUM_FEATURES)
        
        """ 2D, 2 column array that maps each element of the feature_vector
        onto a group and feature of the input. The 0th column holds the 
        feature indices and the first column holds the group indices.
        A group number of -3 refers to sensors, -2 refers to primitives,
        and -1 refers to actions.
        """ 
        self.index_map_inverse = np.zeros((self.MAX_NUM_FEATURES, 2), np.int)

        """ State that provides memory of the input on 
        the previous time step 
        """
        self.previous_input = state.State(num_sensors, num_primitives, 
                                          num_actions)

        """
        self.input_map: State that maps input groups to feature groups
        self.index_map: State that maps input elements to the feature vector
        """
        self.input_map = self.previous_input.zeros_like()
        self.index_map = self.previous_input.zeros_like()
        
        """ Initialize sensor aspects """ 
        self.index_map.sensors = np.cumsum(np.ones(num_sensors, np.int)) - 1
        self.index_map_inverse[ :num_sensors, :] = np.vstack(
                           (np.cumsum( np.ones( num_sensors, np.int)) - 1,
                            -3 * np.ones(num_sensors, np.int))).transpose()
        self.last_entry = num_sensors

        """ Initialize primitive aspects """
        self.index_map.primitives = np.cumsum(np.ones( 
                           num_primitives, np.int)) - 1 + self.last_entry
        self.index_map_inverse[self.last_entry: 
                               self.last_entry + num_primitives, :] = \
                   np.vstack((np.cumsum(np.ones( num_primitives, np.int)) - 1, 
                   -2 * np.ones( num_primitives, np.int))).transpose()
        self.last_entry += num_primitives
        
        """ Initialize action aspects """
        self.index_map.actions = np.cumsum(np.ones( 
                            num_actions, np.int)) - 1 + self.last_entry
        self.index_map_inverse[self.last_entry: 
                               self.last_entry + num_actions, :] = \
                   np.vstack((np.cumsum(np.ones( num_actions, np.int)) - 1, 
                   -1 * np.ones( num_actions, np.int))).transpose()
        self.last_entry += num_actions
        
        
    def add_group(self):
        self.previous_input.add_group()
        self.index_map.add_group()


    def add_feature(self, nth_group):
        
        self.previous_input.add_feature(nth_group)
        self.index_map.add_feature(nth_group, self.last_entry)
        self.index_map_inverse[self.last_entry,:] = np.hstack(
                          (len(self.index_map[nth_group]) - 1, nth_group))
        self.last_entry += 1
            
        """ Disallow building new groups out of members of the new feature
        and any of its group's inputs.
        """
        disallowed_elements = np.zeros(0)
        for group_ctr in range(nth_group - 1):    
            
            """ Find the input features indices from group [group_ctr] that 
            contribute to the nth_group.
            """
            matching_feature_indices = (
                  self.input_map.features[nth_group][:,1] == group_ctr).nonzero()
                  
            """ Find the corresponding elements in the feature vector
            and correlation estimation matrix.
            """
            matching_element_indices = self.index_map.features[group_ctr] \
                  [self.input_map.features[nth_group] \
                  [matching_feature_indices, 0]]
                  
            """ Add these to the set of elements that are not allowed to 
            correlate with the new feature to create new groups.
            """
            disallowed_elements = np.hstack(
                       (disallowed_elements, matching_element_indices.ravel()))
            
        """ Perform the same operation on sensors, primitives, and actions """
        """ Sensors """
        matching_feature_indices = (
                  self.input_map.features[nth_group][:,1] == -3).nonzero()
        matching_element_indices = self.index_map.sensors \
                  [self.input_map.features[nth_group] \
                  [matching_feature_indices, 0]]
        disallowed_elements = np.hstack(
                   (disallowed_elements, matching_element_indices.ravel()))
        
        """ Primitives """
        matching_feature_indices = (
                  self.input_map.features[nth_group][:,1] == -2).nonzero()
        matching_element_indices = self.index_map.primitives \
                  [self.input_map.features[nth_group] \
                  [matching_feature_indices, 0]]
        disallowed_elements = np.hstack(
                   (disallowed_elements, matching_element_indices.ravel()))

        """ Actions """
        matching_feature_indices = (
                  self.input_map.features[nth_group][:,1] == -1).nonzero()
        matching_element_indices = self.index_map.actions \
                  [self.input_map.features[nth_group] \
                  [matching_feature_indices, 0]]
        disallowed_elements = np.hstack(
                   (disallowed_elements, matching_element_indices.ravel()))

        """ Propogate unallowable combinations with inputs """ 
        self.combination[self.last_entry - 1, disallowed_elements] = 0
        self.combination[disallowed_elements, self.last_entry - 1] = 0


    def step(self, sensors, primitives, actions, previous_feature_activity):
        """ Incrementally estimate correlation between inputs 
        and form groups when appropriate
        """

        """ Build the feature vector.
        Combine sensors and primitives with 
        previous_feature_activity to create the full input set.
        """
        new_input = copy.deepcopy(previous_feature_activity)
        new_input.sensors = sensors
        new_input.primitives = primitives
        
        """ It's not yet clear whether this should be included or not """
        new_input.actions = actions
        
        """ Decay previous input and combine it with the new input """
        self.previous_input.decay(1 - self.INPUT_DECAY_RATE)
        new_input = new_input.bounded_sum(self.previous_input)
                    
        """ Update previous input, preparing it for the next iteration """    
        self.previous_input = copy.deepcopy(new_input)
            
        if not self.features_full:
            group_added = self.update_correlation_estimate(new_input)
    
        """ Sort input groups into their feature groups """
        input_group = state.State.zeros_like(new_input)
        input_group.sensors = np.zeros_like(sensors)
        input_group.primitives = primitives
        input_group.actions = actions
        for group_ctr in range(input_group.n_feature_groups() - 1):
            for element_counter in range(self.input_map[group_ctr].shape[0]):
                from_group = self.input_map[group_ctr][element_counter, 1]
                from_feature = self.input_map[group_ctr][element_counter, 0]
                
                if from_group == -3:
                    input_group[group_ctr][element_counter] = \
                                    input.sensors[from_feature]
                elif from_group == -2:
                    input_group[group_ctr][element_counter] = \
                                    input.primitives[from_feature]
                elif from_group == -1:
                    input_group[group_ctr][element_counter] = \
                                    input.actions[from_feature]
                else:
                    input_group[group_ctr][element_counter] = \
                                    input.features[from_group][from_feature]

            #input_group[group_ctr] = input_group[group_ctr].ravel()

            #     # TODO: is this necessary?
            #     if k > 2:
            #         #ensures that the total magnitude of the input features are 
            #         #less than one
            #         input_group[k] = input_group[k].ravel() / np.sqrt( len( input_group[k]))

        return input_group, group_added


    def update_correlation_estimate(self, new_input):
        """ Update an estimate of pseudo-correlation between every
        feature and every other feature, including the sensors, primitives,
        and actions. 
        """

        """ Populate the full feature vector """
        self.feature_vector[self.index_map.sensors] = new_input.sensors
        self.feature_vector[self.index_map.primitives] = new_input.primitives
        self.feature_vector[self.index_map.actions] = new_input.actions
        
        for index in range(new_input.n_feature_groups() - 1):
            self.feature_vector[self.index_map.features[index]] = \
                                new_input.features[index]

        """ Find the upper bound on platicity based on how many groups
        each feature is associated with.
        Then update the platicity of each input to form new associations, 
        incrementally stepping the each combintation's platicity toward 
        its upper bound.
        """
        self.platicity[:self.last_entry, :self.last_entry] += \
            self.PLASTICITY_UPDATE_RATE * \
            (self.MAX_PROPENSITY - \
             self.platicity[:self.last_entry, :self.last_entry])

        """ Decrease the magnitude of features if they are already 
        inputs to feature groups. The penalty is a negative exponential 
        in the number of groups that each feature belongs to. 
        """ 
        weighted_feature_vector = \
            (np.exp( - self.groups_per_feature [:self.last_entry] * \
                     self.GROUP_DISCOUNT ) * \
                     self.feature_vector[:self.last_entry])[np.newaxis]  \
                     # newaxis needed for it to be treated as 2D
        
        """ Determine the pseudo-correlation value according to the 
        only the current inputs. It is the product of every weighted 
        feature activity with every other weighted feature activity.
        """
        instant_correlation = np.dot(weighted_feature_vector.transpose(), 
                     weighted_feature_vector)
        
        """ Determine the upper bound on the size of the incremental step 
        toward the instant pseudo-correlation. It is weighted both by the 
        feature associated with the column of the correlation estimate and
        the platicity of each pair of elements. Weighting by the feature
        column introduces an asymmetry, that is the correlation of feature
        A with feature B is not necessarily the same as the correlation of
        feature B with feature A.
        """
        delta_correlation = np.tile(weighted_feature_vector, \
                     (self.last_entry, 1)) * \
                     (instant_correlation - \
                     self.correlation[:self.last_entry, :self.last_entry])
                     
        """ Adapt correlation toward average activity correlation by
        the calculated step size.
        """
        self.correlation[:self.last_entry, :self.last_entry] += \
                     self.platicity[:self.last_entry, :self.last_entry]* \
                     delta_correlation

        """ Update legal combinations in the correlation matrix """
        self.correlation[:self.last_entry, :self.last_entry] *= \
            self.combination[:self.last_entry, :self.last_entry]

        """ Update the plasticity by subtracting the magnitude of the 
        correlation change. 
        """
        self.platicity[:self.last_entry, :self.last_entry] = \
            np.maximum(self.platicity[:self.last_entry, 
                                       :self.last_entry] - \
            np.abs(delta_correlation), 0)

        """ Check to see whether the capacity to store and update features
        has been reached.
        """
        if self.last_entry > self.MAX_NUM_FEATURES * 0.95:
            self.features_full = True
            print('==Max number of features almost reached (%s)==' 
                  % self.last_entry)

        """ If the correlation is high enough, create a new group """
        group_added = False
        max_correlation = np.max(self.correlation)
        if max_correlation > self.NEW_GROUP_THRESHOLD:
            
            """ Nucleate a new group under the two elements for which 
            correlation is a maximum.
            """
            indices1, indices2 = (self.correlation == 
                                  max_correlation).nonzero()
            which_index = np.random.random_integers(0, len(indices1)-1)
            index1 = indices1[which_index]
            index2 = indices2[which_index]
            added_feature_indices = [index1, index2]

            for element in itertools.product(added_feature_indices, 
                                             added_feature_indices):
                self.correlation[element] = 0
                self.combination[element] = 0

            """ Track the available elements with candidate_matches
            to add and the correlation associated with each.
            """
            candidate_matches = np.zeros(self.combination.shape)
            candidate_matches[:,added_feature_indices] = 1
            candidate_matches[added_feature_indices,:] = 1
            
            """ Add elements one at a time in a greedy fashion until 
            either the maximum 
            number of elements is reached or the minimum amount of
            correlation with the other members of the group cannot be 
            achieved.
            """
            while True:
                """ Calculate overall correaltion of candidate features
                with all the elements already in the group using a combination 
                of column-wise and row-wise correlation values. In true
                correlation, these would be symmetric. In this pseudo-
                correlation, they are not. 
                """
                candidate_correlations = np.abs(self.correlation * 
                                                candidate_matches)
                match_strength_col = np.sum(candidate_correlations, axis=0)
                match_strength_row = np.sum(candidate_correlations, axis=1)
                candidate_match_strength = match_strength_col + \
                                match_strength_row.transpose()
                candidate_match_strength = candidate_match_strength / \
                                (2 * len(added_feature_indices))

                """ Find the next most correlated feature and test 
                whether its correlation is high enough to add it to the
                group.
                """ 
                candidate_match_strength[added_feature_indices] = 0
                if (np.max(candidate_match_strength) < self.MIN_SIG_CORR) \
                        or (len(added_feature_indices) >= self.MAX_GROUP_SIZE):
                    break

                max_match_strength = np.max(candidate_match_strength)
                max_match_strength_indices = (candidate_match_strength == 
                                      max_match_strength).nonzero()[0]
                index = max_match_strength_indices[
                           np.random.random_integers(0, 
                           len(max_match_strength_indices)-1)]
                
                """ Add the new feature """
                added_feature_indices.append(index)
                
                """ Update the 2D arrays that estimate correlation, 
                allowable combinations, and additional candidates for
                the group. """
                for element in itertools.product(added_feature_indices, 
                                                 added_feature_indices):
                    self.correlation [element] = 0
                    self.combination[element] = 0
                candidate_matches[:,added_feature_indices] = 1
                candidate_matches[added_feature_indices,:] = 1

            """ Add the newly-created group """
            added_feature_indices = np.sort(added_feature_indices)
            self.groups_per_feature[added_feature_indices] += 1
            self.input_map.add_group(
                     np.vstack((self.index_map_inverse[index,:] 
                                for index in added_feature_indices)))
            group_added = True
    
        return group_added
    