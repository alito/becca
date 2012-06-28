import copy
from feature_map import FeatureMap
import itertools
import numpy as np
import state
import utils
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
        self.INPUT_DECAY_RATE = 1.0 # real, 0 < x < 1
        
        """ Control how rapidly the coactivity update plasticity changes """
        self.PLASTICITY_UPDATE_RATE = 10. ** (-3) # real, 0 < x < 1, small
        
        """ The maximum value of plasticity """
        self.MAX_PLASTICITY = 0.1
        
        """ The feature actvity penalty associated with 
        prior membership in other groups. 
        """
        self.GROUP_DISCOUNT = 0.5
        
        """ Once a coactivity value exceeds this value, 
        nucleate a new group.
        """ 
        self.NEW_GROUP_THRESHOLD = 0.3     # real,  x >= 0
        
        """ If the coactivity between the first two group members 
        and the next candidates 
        is lower than this value, don't add it. 
        This value approaches NEW_GROUP_THRESHOLD as the group size grows.
        See create_new_group() below.
        """
        self.MIN_SIG_COACTIVITY = 0.2  # real,  x >= self.NEW_GROUP_THRESHOLD
        
        """ The rate that threshold coactivity for adding new
        inputs to the group decays as new inputs are added.
        """
        self.COACTIVITY_THRESHOLD_DECAY_RATE = 0.27 # real, 0 <= x < 1
        
        """ Stop creating new groups, once this number of features 
        is nearly reached.
        """
        self.MAX_NUM_FEATURES = max_num_features
        
        """ Factor controlling the strength of the inhibitory effect 
        that neighboring features have on each other when excersizing
        mutual inhibition.
        """
        self.INHIBITION_STRENGTH_FACTOR = 1.
        
        """ The rate at which features adapt toward observed input data """
        self.FEATURE_ADAPTATION_RATE = 10. ** -1

        """ The rate at which feature fatigue decays.
        0 means it never decays and 1 means it decays immediately--that
        there is no fatigue.
        """
        self.FATIGUE_DECAY_RATE = 10. ** -2
        
        """ The strength of the influence that fatigue has on the
        features.
        """
        self.FATIGUE_SUSCEPTIBILITY = 10. ** -1
        
        """ To prevent normalization from giving a divide by zero """
        self.EPSILON = 1e-6
        
        """ A flag determining whether to stop creating new groups """
        self.features_full = False
        
        """ The list of 2D arrays that translates grouped inputs 
        into feature activities.
        """
        self.feature_map = FeatureMap(num_sensors, num_primitives, 
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
        self.plasticity = np.zeros(
                        (self.MAX_NUM_FEATURES, self.MAX_NUM_FEATURES))
        
        """ 1D array recording the number of groups that each input
        feature is a member of
        """
        self.groups_per_feature = np.zeros((self.MAX_NUM_FEATURES, 1))
        
        """ 1D array containing all the inputs from all the groups.
        Used in order to make updating the coactivity
        less computationally expensive. 
        """
        self.input_activity = np.zeros((self.MAX_NUM_FEATURES, 1))
        
        """ State that provides memory of the input on 
        the previous time step 
        """
        self.previous_input = state.State(num_sensors, num_primitives, 
                                          num_actions)
        
        """ The activity levels of all features in all groups.
        Passed on to the reinforcement learner at each time step.
        """
        self.feature_activity = self.previous_input.zeros_like()
        self.feature_activity.sensors = np.zeros((0,1))

        """ The level of fatigue of each feature in each group """
        self.fatigue = self.feature_activity.zeros_like()
        
        """ 1D arrays that map each element of the input_activity
        onto a group and feature of the input. 
        A group number of -3 refers to sensors, -2 refers to primitives,
        and -1 refers to actions.
        """ 
        self.inv_coactivity_map_group   = np.zeros((self.MAX_NUM_FEATURES, 1), 
                                                    dtype=np.int)
        self.inv_coactivity_map_feature = np.zeros((self.MAX_NUM_FEATURES, 1), 
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
        self.coactivity_map.sensors = np.cumsum(np.ones((num_sensors,1), 
                                                         dtype=np.int), 
                                dtype=np.int, out=np.ones((num_sensors,1))) - 1
                                                 
        self.inv_coactivity_map_group[ :num_sensors, 0] = \
                        -3 * np.ones(num_sensors, dtype=np.int)
        self.inv_coactivity_map_feature[ :num_sensors, 0] = \
                        np.cumsum( np.ones((num_sensors,1), dtype=np.int), 
                                   dtype=np.int) - 1
        self.n_inputs = num_sensors

        """ Initialize primitive aspects """
        self.coactivity_map.primitives = np.cumsum(np.ones( 
                           (num_primitives,1), dtype=np.int), 
                                        dtype=np.int) - 1 + self.n_inputs
        self.inv_coactivity_map_group[self.n_inputs: 
                               self.n_inputs + num_primitives, 0] = \
                        -2 * np.ones(num_primitives, dtype=np.int)
        self.inv_coactivity_map_feature[self.n_inputs: 
                               self.n_inputs + num_primitives, 0] = \
                        np.cumsum( np.ones(num_primitives, dtype=np.int), 
                                   dtype=np.int) - 1
        self.n_inputs += num_primitives
        
        """ Initialize action aspects """
        self.coactivity_map.actions = np.cumsum(np.ones( 
                            (num_actions,1), dtype=np.int), 
                                         dtype=np.int) - 1 + self.n_inputs
        self.inv_coactivity_map_group[self.n_inputs: 
                               self.n_inputs + num_actions, 0] = \
                        -1 * np.ones(num_actions, dtype=np.int)
        self.inv_coactivity_map_feature[self.n_inputs: 
                               self.n_inputs + num_actions, 0] = \
                        np.cumsum( np.ones(num_actions, dtype=np.int), 
                                   dtype=np.int) - 1
        self.n_inputs += num_actions
        
        """ Disallowed co-activities. 
        A list of 1D numpy arrays, one for each feature group, 
        giving the coactivity
        indices that features in that group are descended from. The coactivity
        between these is forced to be zero to discourage these being combined 
        to create new features.
        """
        self.disallowed = []
        
        self.fatigue_history = []
        self.feature_history = []
        self.inhibition_history = []
        

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
            grouped_input.add_fixed_group(
                   self.grouping_map_feature.features[group_index].size)
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

        """ Calculate feature activity """            
        [excitation, scaled_input] = self.calculate_excitation(grouped_input)
        self.calculate_activity(excitation)
        
        """ Evolve features """
        inhibition = self.calculate_inhibition(excitation)
        self.update_features(scaled_input, inhibition)
        
        """ Calculate fatigue """
        self.fatigue = self.fatigue.unbounded_sum(self.feature_activity)
        self.fatigue = self.fatigue.multiply(1 - self.FATIGUE_DECAY_RATE)
        
        '''
        #debug
        if len(self.feature_map.features) > 0:
            if np.random.random_sample() < .01:
    
                #self.make_history(inhibition.features[0], 
                                   self.inhibition_history, 
                                   label='inhibition history')
                #self.make_history(self.fatigue.features[0], 
                                   self.fatigue_history, 
                                   label='fatigue history')
                self.make_history(self.feature_map.features[0], 
                                  self.feature_history, 
                                  label='feature history')
                viz_utils.force_redraw()
        '''    

        return self.feature_activity


    def update_coactivity_matrix(self, new_input):
        """ Update an estimate of coactivity between every
        feature and every other feature, including the sensors, primitives,
        and actions. 
        """

        """ Populate the full feature vector """
        self.input_activity[ \
                self.coactivity_map.sensors.ravel().astype(int)] = \
                new_input.sensors
        self.input_activity[ \
                self.coactivity_map.primitives.ravel().astype(int)] = \
                new_input.primitives
        self.input_activity[ \
                self.coactivity_map.actions.ravel().astype(int)] = \
                new_input.actions
        
        for index in range(new_input.n_feature_groups()):
            if new_input.features[index].size > 0:
                self.input_activity[ \
                    self.coactivity_map.features[index].ravel().astype(int)] = \
                    new_input.features[index]

        """ Find the upper bound on plasticity based on how many groups
        each feature is associated with.
        Then update the plasticity of each input to form new associations, 
        incrementally stepping the each combintation's plasticity toward 
        its upper bound.
        """
        self.plasticity[:self.n_inputs, :self.n_inputs] += \
            self.PLASTICITY_UPDATE_RATE * \
            (self.MAX_PLASTICITY - \
             self.plasticity[:self.n_inputs, :self.n_inputs])

        """ Decrease the magnitude of features if they are already 
        inputs to feature groups. The penalty is a negative exponential 
        in the number of groups that each feature belongs to. 
        """ 
        weighted_feature_vector = \
            (np.exp( - self.groups_per_feature [:self.n_inputs] * \
                     self.GROUP_DISCOUNT ) * \
                     self.input_activity[:self.n_inputs])
        
        """ Determine the coactivity value according to the 
        only the current inputs. It is the product of every weighted 
        feature activity with every other weighted feature activity.
        """
        instant_coactivity = np.dot(weighted_feature_vector, 
                     weighted_feature_vector.transpose())
        
        """ Determine the upper bound on the size of the incremental step 
        toward the instant coactivity. It is weighted both by the 
        feature associated with the column of the coactivity estimate and
        the plasticity of each pair of elements. Weighting by the feature
        column introduces an asymmetry, that is the coactivity of feature
        A with feature B is not necessarily the same as the coactivity of
        feature B with feature A.
        """
        delta_coactivity = np.tile(weighted_feature_vector.transpose(), \
                     (self.n_inputs, 1)) * \
                     (instant_coactivity - \
                     self.coactivity[:self.n_inputs, :self.n_inputs])
                     
        """ Adapt coactivity toward average activity coactivity by
        the calculated step size.
        """
        self.coactivity[:self.n_inputs, :self.n_inputs] += \
                     self.plasticity[:self.n_inputs, :self.n_inputs]* \
                     delta_coactivity

        """ Update legal combinations in the coactivity matrix """
        self.coactivity[:self.n_inputs, :self.n_inputs] *= \
            self.combination[:self.n_inputs, :self.n_inputs]

        """ Update the plasticity by subtracting the magnitude of the 
        coactivity change. 
        """
        self.plasticity[:self.n_inputs, :self.n_inputs] = \
            np.maximum(self.plasticity[:self.n_inputs, 
                                       :self.n_inputs] - \
            np.abs(delta_coactivity), 0)

        return
    
    
    def create_new_group(self):
        """ If the right conditions have been reached,
        create a new group.
        """
    
        """ Check to see whether the capacity to store and update features
        has been reached.
        """
        if self.n_inputs > self.MAX_NUM_FEATURES * 0.95:
            self.features_full = True
            print('==Max number of features almost reached (%s)==' 
                  % self.n_inputs)

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
            coactivity_threshold = self.MIN_SIG_COACTIVITY
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
                if (np.max(candidate_match_strength) < coactivity_threshold):
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
                
                """ Update the coactivity threshold. This helps keep
                the group from becoming too large. This formulation
                results in an exponential decay 
                """
                coactivity_threshold = coactivity_threshold + \
                            self.COACTIVITY_THRESHOLD_DECAY_RATE * \
                            (self.NEW_GROUP_THRESHOLD - coactivity_threshold)

            """ Add the newly-created group.
            Update all the variables that need to grow to represent 
            the new group.
            """
            added_feature_indices = np.sort(added_feature_indices)
            self.groups_per_feature[added_feature_indices] += 1
            
            """ Add a fixed number of features all at once """
            nth_group = self.feature_activity.n_feature_groups()

            n_group_inputs = len(added_feature_indices)
            n_group_features = n_group_inputs + 2

            self.feature_activity.add_fixed_group(n_group_features)
            self.fatigue.add_fixed_group(n_group_features)
            self.previous_input.add_fixed_group(n_group_features)

            new_coactivity_map_array = \
                            np.cumsum(np.ones((n_group_features,1)), axis=0) \
                            + self.n_inputs - 1
            self.coactivity_map.add_fixed_group(n_group_features, \
                            new_array=new_coactivity_map_array, dtype=np.int)

            self.feature_map.add_fixed_group(n_group_features, n_group_inputs)

            self.grouping_map_group.add_fixed_group(n_group_inputs,
                new_array=self.inv_coactivity_map_group \
                [added_feature_indices], dtype=np.int)
            self.grouping_map_feature.add_fixed_group(n_group_inputs,
                new_array=self.inv_coactivity_map_feature \
                [added_feature_indices], dtype=np.int)

            self.inv_coactivity_map_group[self.n_inputs: \
                          self.n_inputs + n_group_features] =  nth_group
            self.inv_coactivity_map_feature[self.n_inputs: \
                          self.n_inputs + n_group_features] = \
                          np.cumsum(np.ones((n_group_features,1)), axis=0) \
                          - np.ones((n_group_features,1))
            self.n_inputs += n_group_features

            """ Calculate the disallowed cominations for the new group """
            nth_group = self.feature_activity.n_feature_groups() - 1;
            disallowed_elements = np.zeros((0,1), dtype=np.int)
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
                    
                    if self.coactivity_map.features[group_ctr].size > 0:
                        matching_element_indices = self.coactivity_map.features \
                              [group_ctr][matching_feature_members,:]

                    else:
                        matching_element_indices = np.zeros((0,1))
                        
                """ Add these to the set of elements that are not allowed to 
                be grouped with the new feature to create new groups.
                """                
                disallowed_elements = np.vstack((disallowed_elements, 
                            matching_element_indices.ravel()[:,np.newaxis]))
                
            self.disallowed.append(disallowed_elements)
            
            """ Disallow building new groups out of members of the new feature
            and any of its group's inputs.
            """
            self.combination[self.n_inputs - n_group_features:self.n_inputs, 
                             self.disallowed[nth_group][:,0].astype(int)] = 0
            self.combination[self.disallowed[nth_group][:,0].astype(int), 
                             self.n_inputs - n_group_features:self.n_inputs] = 0
            
            print 'adding group ', self.previous_input.n_feature_groups() - 1, \
                    ' with ', len(added_feature_indices), ' inputs'

        return 


    def calculate_excitation(self, inputs):
        """ Find the excitation level for each feature """
        scaled_input = inputs.zeros_like()
        excitation = self.feature_activity.zeros_like()        
        excitation.primitives =  \
                copy.deepcopy(inputs.primitives)
        excitation.actions = copy.deepcopy(inputs.actions)
            
        for group_index in range(inputs.n_feature_groups()):
            """ Scaling the input in this way:
            
            scaled_input = input * max(input) / norm(input)
            
            guarantees that it also will have a maximum 
            magnitude of 1. 
            """
            these_inputs = inputs.features[group_index]                 
            scaled_input.features[group_index] = \
                            these_inputs * np.max(these_inputs) / \
                            (np.linalg.norm(these_inputs) + self.EPSILON)

            """ cosine^2 tuning 
            This formulation of excitation was chosen to have the
            property that the maximum excitation possible for
            any given feature is 1. 
            
            excitation  = cos^2(angle between input and feature)
            """
            excitation.features[group_index] = np.dot( \
                     self.feature_map.features[group_index], \
                     scaled_input.features[group_index]) ** 2

        return excitation, scaled_input
    
        
    def calculate_activity(self, excitation):
        """ Find the activity of each feature, after excitation and 
        inhibition. This includes
        1) figuring out which feature wins and
        2) figuring out what the activity magnitude is--for now it's just
        the excitation
        """
        self.feature_activity = excitation.zeros_like()
        self.feature_activity.primitives = excitation.primitives
        #self.feature_activity.actions = excitation.actions
        
        for group_index in range(excitation.n_feature_groups()):
            vote = excitation.features[group_index] * \
                    np.exp(- self.FATIGUE_SUSCEPTIBILITY * \
                           self.fatigue.features[group_index])

            winner = np.argmax(vote)
            self.feature_activity.features[group_index][winner,0] = \
                    excitation.features[group_index][winner,0]
                    
        return 
    
    
    def calculate_inhibition(self, excitation):
        """ Find the extent to which each feature is inhibited by the
        othere features in its group.
        This formulation guarantees that inhibition of each feature will
        be between 0 and 1:
        
        inhibition_i = C * sum(excitation_j * similarity_ij) for all j!=i
        
        where
            inhibition_i is the inhibition of the ith feature
            C is a constant modulating the inhibition strength
            excitation_j is the excitation of the jth feature
            similarity_ij is the similarity between the ith and jth features,
                given by utils.similarity
            
        This quantity can become greater than 1. The final inhibition
        factor, which maps onto (0,1], is given by 
        
        e^(-inhibition_i)
        """
        
        inhibition = excitation.zeros_like()
        
        for group_index in range(excitation.n_feature_groups()):            
            feature_index = \
                    np.argmax(self.feature_activity.features[group_index])
            
            similarities = utils.similarity( 
                self.feature_map.features[group_index][feature_index,:], 
                self.feature_map.features[group_index].transpose())

            cos_theta = \
                    np.dot(self.feature_map.features \
                    [group_index][feature_index,:], \
                    self.feature_map.features[group_index].transpose())
            similarities = cos_theta ** 2
            
            """ Ignore its similarity with itself """
            similarities[feature_index] = 0.
            
            inhibition_strength = np.dot(similarities, \
                                         excitation.features[group_index]) * \
                                         self.INHIBITION_STRENGTH_FACTOR
                
            inhibition.features[group_index][feature_index,0] = \
                                         np.exp(-inhibition_strength)
                
        return inhibition
     
     
    def update_features(self, scaled_input, inhibition):
        """ Make the winning feature migrate toward the input that 
        excited it.
        """
        for group_index in range(inhibition.n_feature_groups()):
            
            """ The feature to update """
            winner = np.flatnonzero(self.feature_activity.features[group_index])
            
            """ The direction in which to update it """
            delta = scaled_input.features[group_index].transpose() - \
                    self.feature_map.features[group_index][winner,:]
                    
            """ How far to update it in that direction """
            step_fraction = \
                    self.feature_activity.features[group_index][winner,:] * \
                    inhibition.features[group_index][winner,:] * \
                    self.FEATURE_ADAPTATION_RATE
            
            """ Perform the update """
            self.feature_map.features[group_index][winner,:] += \
                                                delta * step_fraction
                                                
            """ Renormalize the feature to make sure it has unit 
            magnitude.
            """
            self.feature_map.features[group_index][winner,:] /= \
              np.linalg.norm(self.feature_map.features[group_index][winner,:])
              
        return
       
        
    def size(self):
        """ Determine the approximate number of elements being used by the
        class and its members. Created to debug an apparently excessive 
        use of memory.
        """
        total = 0
        total += self.feature_map.size()
        total += self.coactivity.size
        total += self.combination.size
        total += self.plasticity.size
        total += self.groups_per_feature.size
        total += self.input_activity.size
        total += self.previous_input.size()
        total += self.feature_activity.size()
        total += self.inv_coactivity_map_group.size
        total += self.inv_coactivity_map_feature.size
        total += self.grouping_map_group.size()
        total += self.grouping_map_feature.size()
        total += self.coactivity_map.size()

        return total
            
            
    def visualize(self, save_eps=False):
        viz_utils.visualize_grouper_coactivity(self.coactivity, \
                                          self.n_inputs, save_eps)
        viz_utils.visualize_grouper_hierarchy(self, save_eps)
        #viz_utils.visualize_feature_set(self, save_eps)
        #viz_utils.visualize_feature_spacing(self)
        
        viz_utils.force_redraw()
        return
    
    
    def make_history(self, recordable_array, array_history, label=None):
        array_history.append(copy.deepcopy(recordable_array))
        
        if np.random.random_sample() < 1.:
            viz_utils.visualize_array_list(array_history, label)
