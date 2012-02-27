import copy
import itertools
import logging

import numpy as np

from .. import utils

class Grouper(object):

    def __init__(self, num_sensors, num_actions, num_primitives, max_num_features, graphs=True):

        self.INPUT_DECAY_RATE = 1.0
        self.PROPENSITY_UPDATE_RATE = 10 ** (-3) # real, 0 < x < 1
        self.MAX_PROPENSITY = 0.1
        self.GROUP_DISCOUNT = 0.5
        self.NEW_GROUP_THRESHOLD = 0.3     # real,  x >= 0
        self.MIN_SIG_CORR = 0.05  # real,  x >= 0
        self.MAX_GROUP_SIZE = 100
        self.MAX_NUM_FEATURES = max_num_features

        self.features_full = False

        self.correlation = np.zeros((self.MAX_NUM_FEATURES, self.MAX_NUM_FEATURES))
        self.combination = np.ones( (self.MAX_NUM_FEATURES, self.MAX_NUM_FEATURES)) - np.eye( self.MAX_NUM_FEATURES)
        self.propensity = np.zeros((self.MAX_NUM_FEATURES, self.MAX_NUM_FEATURES))
        self.groups_per_feature = np.zeros( self.MAX_NUM_FEATURES)
        self.feature_vector = np.zeros(self.MAX_NUM_FEATURES)
        self.index_map_inverse = np.zeros((self.MAX_NUM_FEATURES, 2), np.int)

        #self.input_map   #maps input groups to feature groups
        #self.index_map    #maps input groups to correlation indices
        #self.index_map_inverse#maps correlation indices to input groups

        self.previous_input = []
        self.input_map = []
        self.index_map = []
        
        #initializes group 0
        self.previous_input.append(np.zeros( num_sensors))
        self.input_map.append(np.vstack((np.cumsum(np.ones(num_sensors, np.int)) - 1, 
                                         np.zeros(num_sensors, np.int))).transpose())
        self.index_map.append(np.cumsum(np.ones(num_sensors, np.int)) - 1)
        self.index_map_inverse[ :num_sensors, :] = np.vstack(( np.cumsum( np.ones( num_sensors, np.int)) - 1,
                                                               np.zeros(num_sensors, np.int))).transpose()
        self.last_entry = num_sensors

        #initializes group 1
        self.previous_input.append(np.zeros(num_primitives))
        self.input_map.append(np.vstack((np.cumsum(np.ones( num_primitives, np.int)) - 1, 
                                         np.ones( num_primitives, np.int))).transpose())
        self.index_map.append(np.cumsum(np.ones( num_primitives, np.int)) - 1 + self.last_entry)

        self.index_map_inverse[self.last_entry: self.last_entry + num_primitives, :] = \
            np.vstack((np.cumsum(np.ones( num_primitives, np.int)) - 1, np.ones( num_primitives, np.int))).transpose()
        self.last_entry += num_primitives
        
        #initializes group 2
        self.previous_input.append(np.zeros( num_actions))
        self.input_map.append(np.vstack(( np.cumsum(np.ones( num_actions, np.int)) - 1,
                                          2 * np.ones( num_actions, np.int) )).transpose())
        self.index_map.append(np.cumsum( np.ones( num_actions, np.int)) - 1 + self.last_entry)
        self.index_map_inverse[self.last_entry: self.last_entry + num_actions, :] = \
            np.vstack(( np.cumsum( np.ones( num_actions, np.int)) - 1, 2 * np.ones( num_actions, np.int))).transpose()
        self.last_entry += num_actions
       
        self.graphing = graphs

        
    def add_group(self):
        self.previous_input.append(np.zeros(1))


    def add_feature(self, nth_group, has_dummy):
        
        self.previous_input[nth_group] = np.hstack((self.previous_input[nth_group], [0]))

        if has_dummy:
            self.previous_input[nth_group] = self.previous_input[ nth_group][1:]
            self.correlation[self.index_map[nth_group], :] = 0
            self.correlation[:, self.index_map[nth_group]] = 0
            self.propensity[self.index_map[nth_group], :] = 0
            self.propensity[:, self.index_map[nth_group]] = 0
        else:
            self.last_entry += 1
            self.index_map[nth_group] = np.vstack((self.index_map[nth_group], self.last_entry - 1))
            self.index_map_inverse[self.last_entry - 1,:] = np.hstack((self.index_map[nth_group].shape[1], nth_group))


        element_index_correlations = None
        for subindex in range(nth_group):
            indices = ( self.input_map[nth_group][:,1] == subindex).nonzero()[0]
            element_index = self.index_map[subindex][self.input_map[nth_group][indices]]
            if element_index_correlations is None:
                element_index_correlations = element_index.ravel()
            else:
                element_index_correlations = np.hstack((element_index_correlations, element_index.ravel()))


        #propogate unallowable combinations with inputs 
        self.combination[self.last_entry - 1, element_index_correlations] = 0
        self.combination[element_index_correlations, self.last_entry - 1] = 0



    def step(self, sensors, primitives, action, previous_feature_activity):

        # incrementally estimates correlation between inputs and forms groups
        # when appropriate
        num_groups = len(previous_feature_activity)

        # builds the feature vector
        # combines sensors and primitives with 
        # previous_feature_activity to create the full input set
        input = copy.deepcopy(previous_feature_activity)
        
        input[0] = sensors
        input[1] = primitives

        #debug
        input[2] = action

        for index in range(num_groups):
            # decays previous input
            self.previous_input[index] *= (1 - self.INPUT_DECAY_RATE)  

            # adds previous input to input
            input[index] = utils.bounded_sum( input[index], self.previous_input[index])

            # updates previous input, preparing it for the next iteration
            self.previous_input[index] = input[index]

            
        # initializes remainder of entries to the input pool.
        for index in range(num_groups):
            self.feature_vector[self.index_map[index]] = input[index]


        group_added = False

        if not self.features_full:

            # finds the upper bound on propensity based on how many groups
            # each feature is associated with
            # updates the propensity of each input to form new associations

            self.propensity[:self.last_entry, :self.last_entry] += \
                self.PROPENSITY_UPDATE_RATE * \
                (self.MAX_PROPENSITY - self.propensity[:self.last_entry, :self.last_entry])

            # performs clustering on feature_input to create feature groups 
            # adapts correlation toward average activity correlation

            weighted_feature_vector = \
                (np.exp( - self.groups_per_feature [:self.last_entry] * self.GROUP_DISCOUNT ) * \
                    self.feature_vector[:self.last_entry])[np.newaxis]  # newaxis needed for it to be treated as 2D
            
            delta_correlation =   np.tile( weighted_feature_vector, (self.last_entry, 1)) * \
                (np.dot(weighted_feature_vector.transpose(), weighted_feature_vector) - \
                    self.correlation[:self.last_entry, :self.last_entry])
            self.correlation[:self.last_entry, :self.last_entry] += \
                self.propensity[:self.last_entry, :self.last_entry] * delta_correlation

            #updates legal combinations in the correlation matrix
            self.correlation[:self.last_entry, :self.last_entry] *= \
                self.combination[:self.last_entry, :self.last_entry]

            self.propensity[:self.last_entry, :self.last_entry] = \
                np.maximum( self.propensity[:self.last_entry, :self.last_entry] - np.abs(delta_correlation), 0)

            if self.last_entry > self.MAX_NUM_FEATURES * 0.95:
                self.features_full = True
                logging.warn('==Max number of features almost reached (%s)==' % self.last_entry)

            max_correlation = np.max(self.correlation)
            if max_correlation > self.NEW_GROUP_THRESHOLD:

                group_added = True
                indices1, indices2 = (self.correlation == max_correlation).nonzero()
                which_index = np.random.random_integers(0, len(indices1)-1)
                index1 = indices1[which_index]
                index2 = indices2[which_index]
                relevant_indices = [index1, index2]

                for element in itertools.product(relevant_indices, relevant_indices):
                    self.correlation[element] = 0
                    self.combination[element] = 0
                

                # Z tracks the available elements to add and the correlation 
                # with each
                Z = np.zeros( self.combination.shape)

                Z[:,relevant_indices] = 1
                Z[relevant_indices,:] = 1
                while True:
                    T = np.abs( self.correlation * Z)
                    Tc = np.sum(T, axis=0)
                    Tr = np.sum(T, axis=1)
                    Tl = Tc + Tr.transpose()
                    Tl = Tl / (2 * len(relevant_indices))

                    Tl[relevant_indices] = 0
                    if ( np.max(Tl) < self.MIN_SIG_CORR) or (len(relevant_indices) >= self.MAX_GROUP_SIZE):
                        break

                    max_tl = np.max(Tl)
                    max_tl_indices = (Tl == max_tl).nonzero()[0]
                    index = max_tl_indices[np.random.random_integers(0, len(max_tl_indices)-1)]
                    relevant_indices.append(index)

                    for element in itertools.product(relevant_indices, relevant_indices):
                        self.correlation [element] = 0
                        self.combination[element] = 0
                    Z[:,relevant_indices] = 1
                    Z[relevant_indices,:] = 1

                element = np.sort(element)
                self.groups_per_feature[element] += 1

                self.input_map.append(None)
                for index in relevant_indices:
                    if self.input_map[-1] is None:
                        self.input_map[-1] = self.index_map_inverse[index, :]
                    else:
                        self.input_map[-1] = np.vstack((self.input_map[-1], self.index_map_inverse[index, :]))


                #initializes a new group
                self.last_entry += 1
                self.index_map.append(self.last_entry)
                self.index_map_inverse[self.last_entry, :] = np.array([0, num_groups], np.int)

                num_groups += 1
                

        input_group = [utils.empty_array()]
        for index in range(1,num_groups):
            input_group.append(np.zeros(self.input_map[index].shape[0]))
            # sorts inputs into their groups
            for input_counter in range(self.input_map[index].shape[0]):
                input_group[index][input_counter] =  input[ self.input_map[index][input_counter, 1]][self.input_map[index][input_counter, 0]]

            input_group[index] = input_group[index].ravel()

        #     # TODO: is this necessary?
        #     if k > 2:
        #         #ensures that the total magnitude of the input features are 
        #         #less than one
        #         input_group[k] = input_group[k].ravel() / np.sqrt( len( input_group[k]))


        return input_group, group_added
