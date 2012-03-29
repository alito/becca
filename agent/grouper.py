import copy
from feature_map import FeatureMap
import itertools
import numpy as np
import state
import visualizer

class Grouper(object):
    """ The object responsible for assembling inputs of various types,
    determining how to group them, and grouping them at each time step.
    Perform clustering on correlation_vector to create feature groups. 
    """
    
    def __init__(self, num_sensors, num_actions, num_primitives, 
                 max_num_features):

        """ Control how rapidly previous inputs are forgotten """
        self.INPUT_DECAY_RATE = 1.0 # real, 0 < x < 1
        
        """ Control how rapidly the correlation update platicity changes """
        self.PLASTICITY_UPDATE_RATE = 10 ** (-2) # real, 0 < x < 1
        
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
        self.MAX_GROUP_SIZE = 30
        
        """ Stop creating new groups, once this number of features 
        is nearly reached.
        """
        self.MAX_NUM_FEATURES = max_num_features

        """ Constants determining the conditions under which new
        features are created.
        """        
        self.NEW_FEATURE_MARGIN = 0.3
        self.NEW_FEATURE_MIN_SIZE = 0.2

        """ A flag determining whether to stop creating new groups """
        self.features_full = False
        
        """ The list of 2D arrays that translates grouped inputs 
        into feature activities.
        """
        self.feature_map = FeatureMap(num_sensors, num_primitives, num_actions)        
        
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
        self.correlation_vector = np.zeros(self.MAX_NUM_FEATURES)
        
        """ State that provides memory of the input on 
        the previous time step 
        """
        self.previous_input = state.State(num_sensors, num_primitives, 
                                          num_actions)

        """ 1D arrays that map each element of the correlation_vector
        onto a group and feature of the input. 
        A group number of -3 refers to sensors, -2 refers to primitives,
        and -1 refers to actions.
        """ 
        self.inv_corr_matrix_map_group   = np.zeros(self.MAX_NUM_FEATURES)
        self.inv_corrr_matrix_map_feature = np.zeros(self.MAX_NUM_FEATURES)

        """
        input_maps group the inputs into groups 
        self.grouping_map_group: for a given group, lists the group that each of 
            its members belong to
        self.grouping_map_feature: for a given group, lists the feature that 
            each of its members correspond to
        self.corr_matrix_map: State that maps input elements to the feature vector
        """
        self.grouping_map_group = self.previous_input.zeros_like()
        self.grouping_map_feature = self.previous_input.zeros_like()
        self.corr_matrix_map = self.previous_input.zeros_like()
        
        """ Initialize sensor aspects """ 
        self.corr_matrix_map.sensors = np.cumsum(np.ones(num_sensors, np.int)) - 1
        self.inv_corr_matrix_map_group[ :num_sensors] = \
                        -3 * np.ones(num_sensors, np.int)
        self.inv_corrr_matrix_map_feature[ :num_sensors] = \
                        np.cumsum( np.ones( num_sensors, np.int)) - 1
        self.n_inputs = num_sensors

        """ Initialize primitive aspects """
        self.corr_matrix_map.primitives = np.cumsum(np.ones( 
                           num_primitives, np.int)) - 1 + self.n_inputs
        self.inv_corr_matrix_map_group[self.n_inputs: 
                               self.n_inputs + num_primitives] = \
                        -2 * np.ones(num_primitives, np.int)
        self.inv_corrr_matrix_map_feature[self.n_inputs: 
                               self.n_inputs + num_primitives] = \
                        np.cumsum( np.ones( num_primitives, np.int)) - 1
        self.n_inputs += num_primitives
        
        """ Initialize action aspects """
        self.corr_matrix_map.actions = np.cumsum(np.ones( 
                            num_actions, np.int)) - 1 + self.n_inputs
        self.inv_corr_matrix_map_group[self.n_inputs: 
                               self.n_inputs + num_actions] = \
                        -1 * np.ones(num_actions, np.int)
        self.inv_corrr_matrix_map_feature[self.n_inputs: 
                               self.n_inputs + num_actions] = \
                        np.cumsum( np.ones( num_actions, np.int)) - 1
        self.n_inputs += num_actions
        
        
    def add_group(self, n_group_inputs):
        self.previous_input.add_group()
        self.corr_matrix_map.add_group()
        self.feature_map.add_group(n_group_inputs)



    def add_feature(self, nth_group, new_feature):

        self.feature_map.add_feature(nth_group, new_feature)

        self.previous_input.add_feature(nth_group)
        self.corr_matrix_map.add_feature(nth_group, self.n_inputs)
        self.inv_corr_matrix_map_group[self.n_inputs] =  nth_group
        self.inv_corrr_matrix_map_feature[self.n_inputs] = \
                            len(self.corr_matrix_map[nth_group]) - 1
        self.n_inputs += 1
            
        """ Disallow building new groups out of members of the new feature
        and any of its group's inputs.
        """
        disallowed_elements = np.zeros(0)
        for group_ctr in range(-3, nth_group - 1):    
            
            """ Find the input features indices from group [group_ctr] that 
            contribute to the nth_group.
            """
            matching_group_member_indices = (
                     self.grouping_map_group.features[nth_group] == 
                     group_ctr).nonzero()
            matching_feature_members = \
                  self.grouping_map_feature.features[nth_group] \
                  [matching_group_member_indices]
                  
            """ Find the corresponding elements in the feature vector
            and correlation estimation matrix.
            """
            if group_ctr == -3:
                matching_element_indices = self.corr_matrix_map.sensors \
                      [matching_feature_members]
            elif group_ctr == -2:
                matching_element_indices = self.corr_matrix_map.primitives \
                      [matching_feature_members]
            elif group_ctr == -1:
                matching_element_indices = self.corr_matrix_map.actions \
                      [matching_feature_members]
            else:
                matching_element_indices = self.corr_matrix_map.features[group_ctr] \
                      [matching_feature_members]
                  
            """ Add these to the set of elements that are not allowed to 
            correlate with the new feature to create new groups.
            """
            disallowed_elements = np.hstack(
                       (disallowed_elements, matching_element_indices.ravel()))
             
        """ Propogate unallowable combinations with inputs """ 
        self.combination[self.n_inputs - 1, disallowed_elements] = 0
        self.combination[disallowed_elements, self.n_inputs - 1] = 0


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
            new_group_length = self.update_correlation_matrix(new_input)
    
        """ Sort input groups into their feature groups """
        grouped_input = state.State()
        grouped_input.sensors = np.zeros_like(sensors)
        grouped_input.primitives = primitives
        grouped_input.actions = actions
        for group_ctr in range(grouped_input.n_feature_groups()):
            if self.grouping_map_group.features[group_ctr].size > 0:
                grouped_input.add_group(np.zeros(
                       self.grouping_map_feature.features[group_ctr].size))
                for element_counter in range(
                         self.grouping_map_feature.features[group_ctr].size):
                    from_group = self.grouping_map_group.features \
                                    [group_ctr][element_counter]
                    from_feature = self.grouping_map_feature.features \
                                    [group_ctr][element_counter]

                    print from_group
                    print from_feature
                    
                    print group_ctr
                    print element_counter
                    print new_input.sensors[from_feature]
                    print grouped_input.features[group_ctr]
                    print grouped_input.features[group_ctr][element_counter]
                    
                    if from_group == -3:
                        grouped_input.features[group_ctr][element_counter] = \
                                 new_input.sensors[from_feature]
                    elif from_group == -2:
                        grouped_input.features[group_ctr][element_counter] = \
                                 new_input.primitives[from_feature]
                    elif from_group == -1:
                        grouped_input.features[group_ctr][element_counter] = \
                                 new_input.actions[from_feature]
                    else:
                        grouped_input.features[group_ctr][element_counter] = \
                                 new_input.features[from_group][from_feature]

            """ debug
            grouped_input[group_ctr] = grouped_input[group_ctr].ravel()

                 # TODO: is this necessary?
                 if k > 2:
                     #ensures that the total magnitude of the input features are 
                     #less than one
                     grouped_input[k] = grouped_input[k].ravel() / np.sqrt( len( grouped_input[k]))
           """
            
        """ Interprets inputs as features and updates feature map 
        when appropriate.
        """
        feature_activity = self.update_feature_map(grouped_input)

        return feature_activity, new_group_length


    def update_correlation_matrix(self, new_input):
        """ Update an estimate of pseudo-correlation between every
        feature and every other feature, including the sensors, primitives,
        and actions. 
        """

        """ Populate the full feature vector """
        self.correlation_vector[self.corr_matrix_map.sensors] = new_input.sensors
        self.correlation_vector[self.corr_matrix_map.primitives] = new_input.primitives
        self.correlation_vector[self.corr_matrix_map.actions] = new_input.actions
        
        for index in range(new_input.n_feature_groups()):
            if new_input.features[index].size > 0:
                print index
                print new_input.features[index]
                print self.corr_matrix_map.features[index]
                print self.correlation_vector[self.corr_matrix_map.features[index]]
                
                self.correlation_vector[self.corr_matrix_map.features[index]] = \
                                    new_input.features[index]

        """ Find the upper bound on platicity based on how many groups
        each feature is associated with.
        Then update the platicity of each input to form new associations, 
        incrementally stepping the each combintation's platicity toward 
        its upper bound.
        """
        self.platicity[:self.n_inputs, :self.n_inputs] += \
            self.PLASTICITY_UPDATE_RATE * \
            (self.MAX_PROPENSITY - \
             self.platicity[:self.n_inputs, :self.n_inputs])

        """ Decrease the magnitude of features if they are already 
        inputs to feature groups. The penalty is a negative exponential 
        in the number of groups that each feature belongs to. 
        """ 
        weighted_feature_vector = \
            (np.exp( - self.groups_per_feature [:self.n_inputs] * \
                     self.GROUP_DISCOUNT ) * \
                     self.correlation_vector[:self.n_inputs])[np.newaxis]  \
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
                     (self.n_inputs, 1)) * \
                     (instant_correlation - \
                     self.correlation[:self.n_inputs, :self.n_inputs])
                     
        """ Adapt correlation toward average activity correlation by
        the calculated step size.
        """
        self.correlation[:self.n_inputs, :self.n_inputs] += \
                     self.platicity[:self.n_inputs, :self.n_inputs]* \
                     delta_correlation

        """ Update legal combinations in the correlation matrix """
        self.correlation[:self.n_inputs, :self.n_inputs] *= \
            self.combination[:self.n_inputs, :self.n_inputs]

        """ Update the plasticity by subtracting the magnitude of the 
        correlation change. 
        """
        self.platicity[:self.n_inputs, :self.n_inputs] = \
            np.maximum(self.platicity[:self.n_inputs, 
                                       :self.n_inputs] - \
            np.abs(delta_correlation), 0)

        """ Check to see whether the capacity to store and update features
        has been reached.
        """
        if self.n_inputs > self.MAX_NUM_FEATURES * 0.95:
            self.features_full = True
            print('==Max number of features almost reached (%s)==' 
                  % self.n_inputs)

        """ If the correlation is high enough, create a new group """
        new_group_length = -1
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
            #self.corr_matrix_map.add_group()
            self.grouping_map_group.add_group(
                    self.inv_corr_matrix_map_group[added_feature_indices,:])
            self.grouping_map_feature.add_group(
                    self.inv_corrr_matrix_map_feature[added_feature_indices,:])
            """self.grouping_map_group.add_group(np.vstack((
                              self.inv_corr_matrix_map_group[index,:] 
                              for index in added_feature_indices))) 
            self.grouping_map_feature.add_group(np.vstack((
                              self.inv_corrr_matrix_map_feature[index,:] 
                              for index in added_feature_indices))) """
            
            new_group_length = len(added_feature_indices)
            
        return new_group_length

    def update_feature_map(self, grouped_input):
        feature_vote = utils.AutomaticList()
        num_groups = len(grouped_input)
        for index in range(1,num_groups):
            if np.max(self.feature_map.map[index][0,:]) == 0:
                margin = 1
            else:
                similarity_values = utils.similarity( grouped_input[index], self.feature_map.map[index].transpose())
                margin = 1 - np.max(similarity_values)


            # initializes feature_vote for basic features.
            if index  == 1:
                feature_vote[index] = copy.deepcopy(grouped_input[index])
                margin = 0

            # initializes feature_vote for basic motor actions.  
            if index  == 2:
                # makes these all zero
                # actions, even automatic ones, don't originate in this way. 
                feature_vote[index] = np.zeros(len(grouped_input[index]))
                margin = 0


            # Calculates the feature votes for all feature in group 'index'.
            if index > 2:
                if self.feature_map.map[index].shape[0] > 0:
                    # This formulation of voting was chosen to have the
                    # property that if the group's contributing inputs are are 1,
                    # it will result in a feature vote of 1.
                    feature_vote[index] = np.sqrt( np.dot(self.feature_map.map[index] ** 2, grouped_input[index]))

            if  margin > self.NEW_FEATURE_MARGIN and \
                np.max( grouped_input[index]) > self.NEW_FEATURE_MIN_SIZE and not self.grouper.features_full:

                # This formulation of feature creation was chosen to have 
                # the property that all feature magnitudes are 1. In other words, 
                # every feature is a unit vector in the vector space formed by its 
                # group.
                new_feature = grouped_input[index] / np.max( grouped_input[index])    
                feature_vote = self.add_feature(new_feature, index, feature_vote)        


        # TODO: boost winner up closer to 1? This might help numerically propogate 
        # high-level feature activity strength. Otherwise it might attentuate and 
        # disappear for all but the strongest inputs. See related TODO note at end
        # of grouper_step.
        self.feature_activity = utils.winner_takes_all(feature_vote)
        
        return
    
    
    def visualize(self):
        viz = visualizer.Visualizer()
        viz.visualize_grouper_correlation(self.correlation, self.n_inputs)
        viz.visualize_grouper_hierarchy(self)
        return