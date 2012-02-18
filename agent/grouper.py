import numpy as np

from . import utils

class Grouper(object):

    def __init__(self, num_sensors, num_actions, num_primitives, max_num_features):
        
        self.INPUT_DECAY_RATE = 1.0
        self.PROPENSITY_UPDATE_RATE = 10 ** (-3) # real, 0 < x < 1
        self.MAX_PROPENSITY = 0.1
        self.GROUP_DISCOUNT = 0.5
        self.NEW_GROUP_THRESHOLD = 0.3     # real,  x >= 0
        self.MIN_SIG_CORR = 0.05  # real,  x >= 0
        self.MAX_GROUP_SIZE = 100
        self.MAX_NUM_FEATURES = max_num_features

        self.features_full = 0
        self.filename_postfix = '_self.mat'

        self.correlation = np.zeros( self.MAX_NUM_FEATURES)
        self.combination = np.ones( self.MAX_NUM_FEATURES) - np.eye( self.MAX_NUM_FEATURES)
        self.propensity = np.zeros( self.MAX_NUM_FEATURES)
        self.groups_per_feature = np.zeros( self.MAX_NUM_FEATURES)
        self.feature_vector = np.zeros(self.MAX_NUM_FEATURES)
        self.index_map_inv = np.zeros( self.MAX_NUM_FEATURES, 2)

        #self.input_map   #maps input groups to feature groups
        #self.index_map    #maps input groups to correlation indices
        #self.index_map_inv#maps correlation indices to input groups

        self.previous_input = []
        self.input_map = []
        self.index_map = []
        
        #initializes group 1
        self.previous_input.append(np.zeros( num_sensors, 1))
        self.input_map.append(np.hstack((np.cumsum(np.ones(num_sensors)), 
                                         np.ones(num_sensors))))
        self.index_map.append(np.cumsum(np.ones(num_sensors)))
        self.index_map_inv[ :num_sensors, :] = np.hstack(( np.cumsum( np.ones( num_sensors)), np.ones(num_sensors)))
        self.last_entry = num_sensors

        #initializes group 2
        self.previous_input.append(np.zeros(num_primitives))
        self.input_map.append(np.hstack((np.cumsum(np.ones( num_primitives)), 
                                         2 * np.ones( num_primitives))))
        self.index_map.append(np.cumsum(np.ones( num_primitives)) + self.last_entry)
        self.index_map_inv[self.last_entry + 1: self.last_entry + num_primitives, :] = \
            np.hstack((np.cumsum(np.ones( num_primitives)), 
                      2 * np.ones( num_primitives)))
        self.last_entry = num_primitives + self.last_entry

        #initializes group 3
        self.previous_input.append(np.zeros( num_actions))
        self.input_map.append(np.hstack(( np.cumsum(np.ones( num_actions)), 3 * np.ones( num_actions) )))
        self.index_map.append(np.cumsum( np.ones( num_actions)) + self.last_entry)
        self.index_map_inv[self.last_entry + 1: self.last_entry + num_actions, :] = \
            np.hstack(( np.cumsum( np.ones( num_actions)), 3 * np.ones( num_actions)))
        self.last_entry = num_actions + self.last_entry


    def add_group(self):
        self.previous_input.append(np.zeros(1))


    def add_feature(self, nth_group, has_dummy):
        
        self.previous_input[nth_group] = np.vstack((self.previous_input[nth_group], [0]))

        if has_dummy:
            self.previous_input[nth_group] = self.previous_input[ nth_group][1:]

            self.correlation[self.index_map[nth_group], :] = 0
            self.correlation[:, self.index_map[nth_group]] = 0
            self.propensity[self.index_map[nth_group], :] = 0
            self.propensity[:, self.index_map[nth_group]] = 0
        else:
            self.last_entry += 1
            self.index_map[nth_group] = np.vstack((self.index_map[nth_group], self.last_entry))
            self.index_map_inv[self.last_entry,:] = np.hstack((self.index_map[nth_group].shape[1], nth_group))


        lmnt_index_correlation = np.array([])
        for subindex in range(nth_group - 1):
            indices = ( self.input_map[nth_group][:,1] == subindex)
            lmnt_index = self.index_map[subindex][self.input_map[nth_group]][indices]
            lmnt_index_correlation = np.vstack((lmnt_index_correlation, lmnt_index.ravel()))

        lmnt_index_correlation = lmnt_index_correlation.ravel()

        #propogate unallowable combinations with inputs 
        self.combination[self.last_entry, lmnt_index_correlation] = 0
        self.combination[lmnt_index_correlation, self.last_entry] = 0



    def step(self, sensors, primitives, action, previous_primitives):

        # incrementally estimates correlation between inputs and forms groups
        # when appropriate
        num_groups = len(previous_primitives)

        # builds the feature vector
        # combines sensors and primitives with 
        # previous_primitives to create the full input set
        input = previous_primitives.copy()
        
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


        group_added = 0

        if not self.features_full:

            # finds the upper bound on propensity based on how many groups
            # each feature is associated with
            # updates the propensity of each input to form new associations
            self.propensity[:self.last_entry, :self.last_entry] = \
                self.propensity[:self.last_entry, :self.last_entry] + \
                self.PROPENSITY_UPDATE_RATE * \
                (self.MAX_PROPENSITY - self.propensity[:self.last_entry, :self.last_entry])

            # performs clustering on feature_input to create feature groups 
            # adapts correlation toward average activity correlation

            weighted_feature_vector = \
                np.exp( - self.groups_per_feature [:self.last_entry] * self.GROUP_DISCOUNT ) * \
                    self.feature_vector[:self.last_entry]
            delta_correlation =   np.tile( weighted_feature_vector, (1, self.last_entry)) * \
                (np.dot(weighted_feature_vector, np.transpose(weighted_feature_vector)) - \
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
                print('==Max number of features almost reached (%s)==' % self.last_entry)

            if np.max(self.correlation) > self.NEW_GROUP_THRESHOLD:

                group_added = 1
                index1, index2 = np.unravel_index(np.argmax(self.correlation), self.correlation.shape)
                which_index = int( np.random.random_sample() * len(index1))
                index1 = index1[which_index]
                index2 = index2[which_index]
                lmnt = np.array([index1, index2])
                self.correlation[lmnt, lmnt] = 0
                self.combination[lmnt, lmnt] = 0

                # Z tracks the available elements to add and the correlation 
                # with each
                Z = np.zeros( self.combination.shape)
                Z[:,lmnt] = 1
                Z[lmnt,:] = 1

                while True:
                    T = np.abs( self.correlation * Z)
                    Tc = np.sum(T, axis=0)
                    Tr = np.sum(T, axis=1)
                    Tl = Tc + Tr.transpose()
                    Tl = Tl / (2 * len(lmnt))
                    Tl[lmnt] = 0
                    if ( np.max(Tl) < self.MIN_SIG_CORR) or (len(lmnt) >= self.MAX_GROUP_SIZE):
                        break

                    index = np.argmax(Tl)
                    index = index[int(np.random.random_sample() * len(index))]
                    lmnt = np.hstack((lmnt,index))
                    self.correlation [lmnt, lmnt] = 0
                    self.combination[lmnt, lmnt] = 0
                    Z[:,lmnt] = 1
                    Z[lmnt,:] = 1

                lmnt = np.sort(lmnt)
                self.groups_per_feature[lmnt] += 1


                num_groups += 1

                self.input_map[num_groups].append(np.array([]))
                for lmnt_element in lmnt:
                    self.input_map[num_groups] = np.vstack((self.input_map[num_groups], self.index_map_inv[lmnt_element, :]))


                #initializes a new group
                self.last_entry += 1
                self.index_map[ num_groups] = self.last_entry
                self.index_map_inv[self.last_entry, :] = np.array([0, num_groups])


        input_group = np.zeros((num_groups, self.input_map[0].shape[0]))
        for index in range(1,num_groups):
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
