import copy
from feature_map import FeatureMap
import itertools
import numpy as np
import state
import utils
import viz_utils

class Grouper(object):
    """ The object responsible for feature creation. 
    This includes assembling inputs of various types,
    determining how to group them, grouping them at each time step,
    building and updating the feature map, and using it at each time step
    to translate the input into feature activity.
    """
    
    def __init__(self, num_sensors, num_real_primitives, num_actions,
                 max_num_features):

        """ Control how rapidly previous inputs are forgotten """
        self.INPUT_DECAY_RATE = 1.0 # real, 0 < x < 1
        
        """ Control how rapidly the coactivity update platicity changes """
        self.PLASTICITY_UPDATE_RATE = 2 * 10 ** (-3) # real, 0 < x < 1, small
        
        """ The maximum value of platicity """
        self.MAX_PROPENSITY = 0.1
        
        """ The feature actvity penalty associated with 
        prior membership in other groups. 
        """
        self.GROUP_DISCOUNT = 0.5
        
        """ Once a coactivity value exceeds this value, 
        nucleate a new group.
        """ 
        self.NEW_GROUP_THRESHOLD = 0.3     # real,  x >= 0
        
        """ Once coactivity between group members and new candidates 
        fall below this value, stop adding them.
        """
        self.MIN_SIG_COACTIVITY = 0.05  # real,  x >= 0
        
        """ Stop growing a group, once it reaches this size """
        self.MAX_GROUP_SIZE = 10 ** 6 # effectively no limit
        
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
        self.feature_map = FeatureMap(num_sensors, num_real_primitives, 
                                      num_actions)        
        
        """ 2D array for holding the estimate of the coactivity """
        self.coactivity = np.zeros(
                        (self.MAX_NUM_FEATURES, self.MAX_NUM_FEATURES))

        """ 2D array for tracking the allowable 
        combinations of elements 
        """
        self.combination = np.ones(
                        (self.MAX_NUM_FEATURES, self.MAX_NUM_FEATURES)) - \
                        np.eye( self.MAX_NUM_FEATURES)
        
        """ 2D array for tracking the propensity for individual
        coactivity values to change, i.e. their plasticty
        """
        self.platicity = np.zeros(
                        (self.MAX_NUM_FEATURES, self.MAX_NUM_FEATURES))
        
        """ 1D array recording the number of groups that each input
        feature is a member of
        """
        self.groups_per_feature = np.zeros(self.MAX_NUM_FEATURES)
        
        """ 1D array containing all the inputs from all the groups.
        Used in order to make updating the coactivity
        less computationally expensive. 
        """
        self.input_activity = np.zeros(self.MAX_NUM_FEATURES)
        
        """ State that provides memory of the input on 
        the previous time step 
        """
        self.previous_input = state.State(num_sensors, num_real_primitives, 
                                          num_actions)
        
        """ The activity levels of all features in all groups.
        Passed on to the reinforcement learner at each time step.
        """
        self.feature_activity = self.previous_input.zeros_like()

        """ 1D arrays that map each element of the input_activity
        onto a group and feature of the input. 
        A group number of -3 refers to sensors, -2 refers to primitives,
        and -1 refers to actions.
        """ 
        self.inv_coactivity_map_group   = np.zeros(self.MAX_NUM_FEATURES, 
                                                    dtype=np.int)
        self.inv_coactivity_map_feature = np.zeros(self.MAX_NUM_FEATURES, 
                                                     dtype=np.int)

        """ input_maps group the inputs into groups 
        self.grouping_map_group: for a given group, 
            lists the group that each of its members belong to.
        self.grouping_map_feature: for a given group, lists the feature that 
            each of its members correspond to.
        self.coactivity_map: State that maps input elements 
            to the feature vector.
        """
        self.grouping_map_group = self.previous_input.zeros_like(dtype=np.int)
        self.grouping_map_feature = self.previous_input.zeros_like(dtype=np.int)
        self.coactivity_map = self.previous_input.zeros_like(dtype=np.int)
        
        """ Initialize sensor aspects """ 
        self.coactivity_map.sensors = np.cumsum(np.ones(num_sensors, 
                                                         dtype=np.int), 
                                                 dtype=np.int) - 1
        self.inv_coactivity_map_group[ :num_sensors] = \
                        -3 * np.ones(num_sensors, dtype=np.int)
        self.inv_coactivity_map_feature[ :num_sensors] = \
                        np.cumsum( np.ones( num_sensors, dtype=np.int), 
                                   dtype=np.int) - 1
        self.n_transitions = num_sensors

        """ Initialize primitive aspects """
        self.coactivity_map.primitives = np.cumsum(np.ones( 
                           num_real_primitives, dtype=np.int), 
                                        dtype=np.int) - 1 + self.n_transitions
        self.inv_coactivity_map_group[self.n_transitions: 
                               self.n_transitions + num_real_primitives] = \
                        -2 * np.ones(num_real_primitives, dtype=np.int)
        self.inv_coactivity_map_feature[self.n_transitions: 
                               self.n_transitions + num_real_primitives] = \
                        np.cumsum( np.ones( num_real_primitives, dtype=np.int), 
                                   dtype=np.int) - 1
        self.n_transitions += num_real_primitives
        
        """ Initialize action aspects """
        self.coactivity_map.actions = np.cumsum(np.ones( 
                            num_actions, dtype=np.int), 
                                         dtype=np.int) - 1 + self.n_transitions
        self.inv_coactivity_map_group[self.n_transitions: 
                               self.n_transitions + num_actions] = \
                        -1 * np.ones(num_actions, dtype=np.int)
        self.inv_coactivity_map_feature[self.n_transitions: 
                               self.n_transitions + num_actions] = \
                        np.cumsum( np.ones( num_actions, dtype=np.int), 
                                   dtype=np.int) - 1
        self.n_transitions += num_actions
        
        """ Disallowed co-activities. 
        A list of 1D numpy arrays, one for each feature group, 
        giving the coactivity
        indices that features in that group are descended from. The coactivity
        between these is forced to be zero to discourage these being combined 
        to create new features.
        """
        self.disallowed = []
        

    def step(self, sensors, primitives, actions):
        """ Incrementally estimate coactivity between inputs 
        and form groups when appropriate
        """

        """ Build the feature vector.
        Combine sensors and primitives with 
        previous feature_activity to create the full input set.
        """
        new_input = copy.deepcopy(self.feature_activity)
        new_input.sensors = sensors
        new_input.primitives = primitives
        
        """ It's not yet clear whether this should be included or not """
        # debug - don't treat actions as primitive inputs
        #new_input.actions = actions
        
        """ Decay previous input and combine it with the new input """
        self.previous_input.decay(1 - self.INPUT_DECAY_RATE)
        new_input = new_input.bounded_sum(self.previous_input)
                    
        """ Update previous input, preparing it for the next iteration """    
        self.previous_input = copy.deepcopy(new_input)

        """ As appropriate, update the coactivity estimate and 
        create new groups.
        """            
        if not self.features_full:
            self.update_coactivity_matrix(new_input)
            self.create_new_group()

        """ Sort input groups into their feature groups """
        grouped_input = state.State()
        grouped_input.sensors = None
        grouped_input.primitives = primitives
        # debug - don't treat actions as primitive inputs
        #grouped_input.actions = actions
        grouped_input.actions = np.zeros(actions.shape)
        for group_index in range(self.grouping_map_group.n_feature_groups()):
            grouped_input.add_group(np.zeros(
                   self.grouping_map_feature.features[group_index].size))
            for input_element_index in range(
                     self.grouping_map_feature.features[group_index].size):
                from_group = self.grouping_map_group.features \
                                [group_index][input_element_index]
                from_feature = self.grouping_map_feature.features \
                                [group_index][input_element_index]

                if from_group == -3:
                    grouped_input.features[group_index][input_element_index] = \
                             new_input.sensors[from_feature]
                elif from_group == -2:
                    grouped_input.features[group_index][input_element_index] = \
                             new_input.primitives[from_feature]
                elif from_group == -1:
                    grouped_input.features[group_index][input_element_index] = \
                             new_input.actions[from_feature]
                else:
                    grouped_input.features[group_index][input_element_index] = \
                             new_input.features[from_group][from_feature]

            """ debug
            grouped_input[group_index] = grouped_input[group_index].ravel()

                 # TODO: is this necessary?
                 if k > 2:
                     #ensures that the total magnitude of the input features are 
                     #less than one
                     grouped_input[k] = grouped_input[k].ravel() / \
                             np.sqrt( len( grouped_input[k]))
           """
            
        """ debug """      
        #if np.random.random_sample() < 0.01:  
            #viz_utils.visualize_state(grouped_input)
            #viz_utils.visualize_feature_set(self, save_eps=True)
            #viz_utils.force_redraw()
            
        """ Updates feature map when appropriate """
        self.update_feature_map(grouped_input)
        
        """ Interprets inputs as features """
        self.get_feature_activity(grouped_input) 
        
        return self.feature_activity


    def update_coactivity_matrix(self, new_input):
        """ Update an estimate of coactivity between every
        feature and every other feature, including the sensors, primitives,
        and actions. 
        """

        """ Populate the full feature vector """
        self.input_activity[self.coactivity_map.sensors] = new_input.sensors
        self.input_activity[self.coactivity_map.primitives] = \
                                                        new_input.primitives
        self.input_activity[self.coactivity_map.actions] = new_input.actions
        
        for index in range(new_input.n_feature_groups()):
            if new_input.features[index].size > 0:
                self.input_activity[self.coactivity_map.features[index]] = \
                                    new_input.features[index]

        """ Find the upper bound on platicity based on how many groups
        each feature is associated with.
        Then update the platicity of each input to form new associations, 
        incrementally stepping the each combintation's platicity toward 
        its upper bound.
        """
        self.platicity[:self.n_transitions, :self.n_transitions] += \
            self.PLASTICITY_UPDATE_RATE * \
            (self.MAX_PROPENSITY - \
             self.platicity[:self.n_transitions, :self.n_transitions])

        """ Decrease the magnitude of features if they are already 
        inputs to feature groups. The penalty is a negative exponential 
        in the number of groups that each feature belongs to. 
        """ 
        weighted_feature_vector = \
            (np.exp( - self.groups_per_feature [:self.n_transitions] * \
                     self.GROUP_DISCOUNT ) * \
                     self.input_activity[:self.n_transitions])[np.newaxis]  \
                     # newaxis needed for it to be treated as 2D
        
        """ Determine the coactivity value according to the 
        only the current inputs. It is the product of every weighted 
        feature activity with every other weighted feature activity.
        """
        instant_coactivity = np.dot(weighted_feature_vector.transpose(), 
                     weighted_feature_vector)
        
        """ Determine the upper bound on the size of the incremental step 
        toward the instant coactivity. It is weighted both by the 
        feature associated with the column of the coactivity estimate and
        the platicity of each pair of elements. Weighting by the feature
        column introduces an asymmetry, that is the coactivity of feature
        A with feature B is not necessarily the same as the coactivity of
        feature B with feature A.
        """
        delta_coactivity = np.tile(weighted_feature_vector, \
                     (self.n_transitions, 1)) * \
                     (instant_coactivity - \
                     self.coactivity[:self.n_transitions, :self.n_transitions])
                     
        """ Adapt coactivity toward average activity coactivity by
        the calculated step size.
        """
        self.coactivity[:self.n_transitions, :self.n_transitions] += \
                     self.platicity[:self.n_transitions, :self.n_transitions]* \
                     delta_coactivity

        """ Update legal combinations in the coactivity matrix """
        self.coactivity[:self.n_transitions, :self.n_transitions] *= \
            self.combination[:self.n_transitions, :self.n_transitions]

        """ Update the plasticity by subtracting the magnitude of the 
        coactivity change. 
        """
        self.platicity[:self.n_transitions, :self.n_transitions] = \
            np.maximum(self.platicity[:self.n_transitions, 
                                       :self.n_transitions] - \
            np.abs(delta_coactivity), 0)

        return
    
    
    def create_new_group(self):
        """ If the right conditions have been reached,
        create a new group.
        """
    
        """ Check to see whether the capacity to store and update features
        has been reached.
        """
        if self.n_transitions > self.MAX_NUM_FEATURES * 0.95:
            self.features_full = True
            print('==Max number of features almost reached (%s)==' 
                  % self.n_transitions)

        """ If the coactivity is high enough, create a new group """
        max_coactivity = np.max(self.coactivity)
        if max_coactivity > self.NEW_GROUP_THRESHOLD:
            
            """ Nucleate a new group under the two elements for which 
            coactivity is a maximum.
            """
            indices1, indices2 = (self.coactivity == 
                                  max_coactivity).nonzero()
            which_index = np.random.random_integers(0, len(indices1)-1)
            index1 = indices1[which_index]
            index2 = indices2[which_index]
            added_feature_indices = [index1, index2]

            for element in itertools.product(added_feature_indices, 
                                             added_feature_indices):
                self.coactivity[element] = 0
                self.combination[element] = 0

            """ Track the available elements with candidate_matches
            to add and the coactivity associated with each.
            """
            candidate_matches = np.zeros(self.combination.shape)
            candidate_matches[:,added_feature_indices] = 1
            candidate_matches[added_feature_indices,:] = 1
            
            """ Add elements one at a time in a greedy fashion until 
            either the maximum 
            number of elements is reached or the minimum amount of
            coactivity with the other members of the group cannot be 
            achieved.
            """
            while True:
                """ Calculate overall coactivity of candidate features
                with all the elements already in the group using a combination 
                of column-wise and row-wise coactivity values. In true
                coactivity, these would be symmetric. In this 
                coactivity, they are not. 
                """
                candidate_coactivities = np.abs(self.coactivity * 
                                                candidate_matches)
                match_strength_col = np.sum(candidate_coactivities, axis=0)
                match_strength_row = np.sum(candidate_coactivities, axis=1)
                candidate_match_strength = match_strength_col + \
                                match_strength_row.transpose()
                candidate_match_strength = candidate_match_strength / \
                                (2 * len(added_feature_indices))

                """ Find the next most correlated feature and test 
                whether its coactivity is high enough to add it to the
                group.
                """ 
                candidate_match_strength[added_feature_indices] = 0
                if (np.max(candidate_match_strength) < 
                    self.MIN_SIG_COACTIVITY) \
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
                
                """ Update the 2D arrays that estimate coactivity, 
                allowable combinations, and additional candidates for
                the group. """
                for element in itertools.product(added_feature_indices, 
                                                 added_feature_indices):
                    self.coactivity [element] = 0
                    self.combination[element] = 0
                candidate_matches[:,added_feature_indices] = 1
                candidate_matches[added_feature_indices,:] = 1

            """ Add the newly-created group.
            Update all the variables that need to grow to represent 
            the new group.
            """
            added_feature_indices = np.sort(added_feature_indices)
            self.groups_per_feature[added_feature_indices] += 1
            
            self.feature_activity.add_group()
            self.previous_input.add_group()
            self.coactivity_map.add_group(dtype=np.int)
            self.feature_map.add_group(len(added_feature_indices))
            self.grouping_map_group.add_group(
                    self.inv_coactivity_map_group[added_feature_indices],
                    dtype=np.int)
            self.grouping_map_feature.add_group(
                    self.inv_coactivity_map_feature[added_feature_indices],
                    dtype=np.int)
            
            """ Calculate the disallowed cominations for the new group """
            nth_group = self.feature_activity.n_feature_groups() - 1;
            disallowed_elements = np.zeros(0, dtype=np.int)
            for group_ctr in range(-3, nth_group):    
                
                """ Find the input features indices from group [group_ctr] that 
                contribute to the nth_group.
                """
                matching_group_member_indices = (
                         self.grouping_map_group.features[nth_group] == 
                         group_ctr).nonzero()[0]
    
                matching_feature_members = \
                      self.grouping_map_feature.features[nth_group] \
                      [matching_group_member_indices]
                      
                """ Find the corresponding elements in the feature vector
                and coactivity estimation matrix.
                """
                if group_ctr == -3:
                    matching_element_indices = self.coactivity_map.sensors \
                          [matching_feature_members]
                elif group_ctr == -2:
                    matching_element_indices = self.coactivity_map.primitives \
                          [matching_feature_members]
                elif group_ctr == -1:
                    matching_element_indices = self.coactivity_map.actions \
                          [matching_feature_members]
                else:
                    matching_element_indices = self.coactivity_map.features \
                          [group_ctr][matching_feature_members]
                      
                """ Add these to the set of elements that are not allowed to 
                be grouped with the new feature to create new groups.
                """
                disallowed_elements = np.hstack((disallowed_elements, 
                                    matching_element_indices.ravel()))
                if group_ctr >= 0:
                    disallowed_elements = np.hstack((disallowed_elements, 
                                    self.disallowed[group_ctr]))

            self.disallowed.append(disallowed_elements)
    
            
            '''
            """ Visualize the new group """
            viz = visualizer.Visualizer()
            # TODO convert arguments to list
            label = str(self.previous_input.n_feature_groups() - 1)
            label = 'latest group'
            viz.visualize_feature(self, 
                   self.inv_coactivity_map_group[added_feature_indices], 
                   self.inv_coactivity_map_feature[added_feature_indices],
                   label)
        
            utils.force_redraw()
            '''
            
            print 'adding group ', self.previous_input.n_feature_groups() - 1, \
                    ' with ', len(added_feature_indices), ' inputs'

        return 


    def update_feature_map(self, grouped_input):

        """ Check whether to add new features to each of the groups """
        for group_index in range(grouped_input.n_feature_groups()):
            
            if self.feature_map.features[group_index].size == 0:
                """ If there are no features in the group yet, set the 
                threshold margin to accept any new feature.
                """
                margin = 1
            else:
                """ Otherwise, set the threshold margin based on how 
                different the closest existing feature is from the current 
                input.
                """
                similarity_values = utils.similarity( \
                         grouped_input.features[group_index], 
                         self.feature_map.features[group_index].transpose())
                margin = 1 - np.max(similarity_values)
            
            if  margin > self.NEW_FEATURE_MARGIN and \
                np.max(grouped_input.features[group_index]) > \
                       self.NEW_FEATURE_MIN_SIZE and \
                not self.features_full:
                """ If all the conditions are met, add the new feature 
                to the group.
                """
                """ This formulation of feature creation was chosen to have 
                the property that all feature magnitudes are 1. In other words, 
                every feature is a unit vector in the vector space formed by its 
                group members.
                """
                new_feature = grouped_input.features[group_index] / \
                        np.linalg.norm( grouped_input.features[group_index])    
                
                self.add_feature(group_index, new_feature) 
                
        return
    
    
    def add_feature(self, nth_group, new_feature):
        
        self.feature_added = nth_group
        
        self.feature_map.add_feature(nth_group, new_feature)

        self.feature_activity.add_feature(nth_group)
        self.previous_input.add_feature(nth_group)
        self.coactivity_map.add_feature(nth_group, self.n_transitions)
        self.inv_coactivity_map_group[self.n_transitions] =  nth_group
        self.inv_coactivity_map_feature[self.n_transitions] = \
                            len(self.coactivity_map.features[nth_group]) - 1
        self.n_transitions += 1
        
        """ Disallow building new groups out of members of the new feature
        and any of its group's inputs.
        """
        self.combination[self.n_transitions - 1, 
                         self.disallowed[nth_group]] = 0
        self.combination[self.disallowed[nth_group], 
                         self.n_transitions - 1] = 0
        
        return


    def get_feature_activity(self, grouped_input):
        
        """ Initialize feature_vote for primitives """
        self.feature_activity.sensors = np.zeros((0,0))
        self.feature_activity.primitives =  \
                copy.deepcopy(grouped_input.primitives)
        self.feature_activity.actions = copy.deepcopy(grouped_input.actions)
            
        for group_index in range(grouped_input.n_feature_groups()):
            if self.feature_activity.features[group_index].size > 0:
                """ This formulation of voting was chosen to have the
                property that if the group's contributing inputs are are 1,
                it will result in a feature vote of 1.
                """
                feature_vote = np.sqrt(np.dot( \
                         self.feature_map.features[group_index] ** 2, \
                         grouped_input.features[group_index]))

                """ TODO: boost winner up closer to 1? This might help 
                numerically propogate high-level feature activity strength. 
                Otherwise it might attentuate and disappear for all but 
                the strongest inputs. See related TODO note at end
                of grouper.step().
                """
                self.feature_activity.features[group_index] = \
                        utils.winner_takes_all(feature_vote)

        return 

    
    def visualize(self, save_eps=False):
        viz_utils.visualize_grouper_coactivity(self.coactivity, \
                                          self.n_transitions, save_eps)
        viz_utils.visualize_grouper_hierarchy(self, save_eps)
        viz_utils.visualize_feature_set(self, save_eps)
        
        viz_utils.force_redraw()
        return