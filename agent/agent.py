'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''

import cPickle as pickle
import copy
import logging

import numpy as np

from .feature_map import FeatureMap
from .planner import Planner
from .model import Model
from .grouper import Grouper
from . import utils

class Agent(object):
    '''A general reinforcement learning agent, modeled on human neurophysiology 
    and performance.  It takes in a time series of 
    sensory input vectors and a scalar reward and puts out a time series 
    of motor commands.  
    New features are created as necessary to adequately represent the data.
    '''

    def __init__(self, num_sensors, num_primitives, num_actions, max_number_features):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(self.__class__.__name__)


        self.timestep = 0
        
        self.num_sensors = num_sensors
        self.num_primitives = num_primitives
        self.num_actions = num_actions

        self.actions = np.zeros(self.num_actions)

        self.GOAL_DECAY_RATE = 0.05   # real, 0 < x <= 1
        self.STEP_DISCOUNT = 0.5      # real, 0 < x <= 1

        # Rates at which the feature activity and working memory decay.
        # Setting these equal to 1 is the equivalent of making the Markovian 
        # assumption about the task--that all relevant information is captured 
        # adequately in the current state.
        self.WORKING_MEMORY_DECAY_RATE = 0.4      # real, 0 < x <= 1
        # also check out self.grouper.INPUT_DECAY_RATE, set in grouper_initialize

        
        # Rates at which the feature activity and working memory decay.
        # Setting these equal to 1 is the equivalent of making the Markovian 
        # assumption about the task--that all relevant information is captured 
        # adequately in the current state.
        self.WORKING_MEMORY_DECAY_RATE = 0.4      # real, 0 < x <= 1
        # also check out self.grouper.INPUT_DECAY_RATE, set in grouper_initialize

        self.grouper = Grouper( num_sensors, num_actions, num_primitives, max_number_features)
        self.feature_map = FeatureMap(num_sensors, num_primitives, num_actions)        
        self.model = Model( num_primitives, num_actions)
        self.planner = Planner(num_actions)

        self.step_counter = 0
        self.num_groups = 3
        self.feature_added = None
        self.debug = False


        # The first group is dedicated to raw sensor information. None of it is
        # passed on directly as features. It must be correlated and combined before
        # it can emerge as part of a feature. As a result, most of the variables
        # associated with this group are empty
        self.feature_activity = [np.array([])]

        # The second group is dedicated to basic features.
        self.feature_activity.append(np.zeros( self.num_primitives))

        # The third group is dedicated to basic actions.
        self.feature_activity.append(np.zeros( self.num_actions))

        # Initializes other variables containing the full feature information.
        self.attended_feature = copy.deepcopy(self.feature_activity)
        self.working_memory = copy.deepcopy(self.feature_activity)
        self.previous_working_memory = copy.deepcopy(self.feature_activity)
        self.goal = copy.deepcopy(self.feature_activity)

        self.feature_stimulation = []
        
        self.action = np.zeros( self.num_actions)


        

    def load(self, pickle_filename):
        loaded = False
        try:
            with open(pickle_filename, 'rb') as agent_data:
                self = pickle.load(agent_data)

            self.logger = logging.getLogger(self.__class__.__name__)
            self.model.create_logger()
            print('Agent restored at timestep ' + str(self.timestep))

        # otherwise initializes from scratch     
        except IOError:
            # world not found
            self.logger.warn("Couldn't open %s for loading" % pickle_filename)
        except pickle.PickleError, e:
            self.logger.error("Error unpickling world: %s" % e)
        else:
            loaded = True

        return loaded

    
    def save(self, pickle_filename):        
        success = False
        try:
            with open(pickle_filename, 'wb') as agent_data:
                # HACK: unset the logger before pickling and restore afterwards since it can't be pickled                
                logger = self.logger
                del self.logger
                model_logger = self.model.logger
                del self.model.logger
                pickle.dump(self, agent_data)

                self.logger = logger
                self.model.logger = model_logger
            self.logger.info('agent data saved at ' + str(self.timestep) + ' time steps')

        except IOError as err:
            self.logger.error('File error: ' + str(err) + ' encountered while saving agent data')
        except pickle.PickleError as perr: 
            self.logger.error('Pickling error: ' + str(perr) + ' encountered while saving agent data')        
        else:
            success = True
            
        return success


    def integrate_state(self, previous_state, new_state, decay_rate):
        num_groups = len(new_state)
        integrated_state = utils.AutomaticList()
        for index in range(1, num_groups):
            integrated_state[index] = utils.bounded_sum(previous_state[index] * (1 - decay_rate), new_state[index])

        return integrated_state
            

    def add_group(self, group_length):
        """
        adds a new group to the agent
        """
        
        self.num_groups += 1
        self.feature_stimulation.append(np.zeros(1))
        self.feature_activity.append(np.zeros(1))
        self.working_memory.append(np.zeros(1))
        self.previous_working_memory.append(np.zeros(1))
        self.attended_feature.append(np.zeros(1))
        self.goal.append(np.zeros(1))

        self.feature_map.add_group(group_length)
        self.grouper.add_group()
        self.model.add_group()
        # self.planner.add_group()


    def add_feature(self, new_feature, nth_group, feature_vote):
        
        has_dummy = np.max(self.feature_map.map[nth_group] [0,:]) == 0
        self.feature_added = 1

        feature_vote[nth_group] = np.vstack((feature_vote[nth_group], 0))
        self.feature_stimulation[nth_group] = np.vstack((self.feature_stimulation[nth_group], 0))
        self.feature_activity[nth_group] = np.vstack((self.feature_activity[nth_group], 0))
        self.working_memory[nth_group] = np.vstack((self.working_memory[nth_group], 0))
        self.previous_working_memory[nth_group] = np.vstack((self.previous_working_memory[nth_group], 0))
        self.attended_feature[nth_group] = np.vstack((self.attended_feature[nth_group], 0))
        self.goal[nth_group] = np.vstack((self.goal[nth_group], 0))

        # if dummy feature is still in place, removes it
        if has_dummy:
            feature_vote[nth_group] = feature_vote[ nth_group][1:]
            self.feature_stimulation[nth_group] = self.feature_stimulation[ nth_group][1:]
            self.feature_activity[nth_group] = self.feature_activity[ nth_group][1:]
            self.working_memory[nth_group] = self.working_memory[ nth_group][1:]
            self.previous_working_memory[nth_group] = self.previous_working_memory[ nth_group][1:]
            self.attended_feature[nth_group] = self.attended_feature[ nth_group][1:]
            self.goal[nth_group] = self.goal[ nth_group][1:]

        self.grouper.add_feature(nth_group, has_dummy)
        self.feature_map.add_feature(nth_group, has_dummy, new_feature)
        self.model.add_feature(nth_group, has_dummy)
        # self.planner.add_feature(nth_group, has_dummy)

        return feature_vote


    def attend(self):
        """
        Selects a feature from feature_activity to attend to.
        """

        self.SALIENCE_NOISE = 0.1

        max_salience_value = 0
        max_salience_group = 0
        max_salience_index = 0

        # salience is a combination of feature activity magnitude, goal magnitude,
        # and a small amount of noise

        salience = utils.AutomaticList()
        for index in range(len(self.feature_activity)):
            count = len( self.feature_activity[index])
            self.attended_feature[index] = np.zeros(count)

            if count > 0:
                # no point in doing this if the feature is empty
                
                salience[index] = self.SALIENCE_NOISE * np.random.random_sample(count)                
                salience[index] += self.feature_activity[index] * (1 + self.goal[index])

                # Picks the feature with the greatest salience.
                max_value = np.max(salience[index])
                max_index = np.argmax(salience[index])


                if max_value >= max_salience_value:
                    max_salience_value = max_value
                    max_salience_group = index
                    max_salience_index = max_index


            if self.planner.act:
                if np.count_nonzero(self.planner.action):
                    deliberate_action_index = self.planner.action.nonzero()

                    #ensures that exploratory actions will be attended
                    max_salience_value = 10
                    max_salience_group = 2
                    max_salience_index = deliberate_action_index

                #self.planner.explore = False  # this doesn't seem to be used anywhere on becca_m

        # debug
        if self.debug:
            print('attended--group: %s, feature: %s, value: %s' % \
                      (max_salience_group, max_salience_index, max_salience_value))


        self.attended_feature[max_salience_group][max_salience_index] = 1


    def update_feature_map(self, grouped_input):
        self.feature_added = 0

        feature_vote = utils.AutomaticList()
        num_groups = len(grouped_input)
        for index in range(1,num_groups):

            if np.max(self.feature_map.map[index][0,:]) == 0:
                margin = 1
            else:

                similarity_values = utils.similarity( grouped_input[index], self.feature_map.map[index].transpose(), 
                    range(len(self.feature_map.map[index])) )
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

            if (( margin > self.feature_map.NEW_FEATURE_MARGIN) and
                (np.max( grouped_input[index]) > self.feature_map.NEW_FEATURE_MIN_SIZE) and self.grouper.features_full):

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


    def expand(self, feature):
        """
        Expands higher-level features into basic features and raw sensory inputs.
        This allows the feature to be expressed in terms of the sensory
        information that activates it. It is conceptually similar to the retinal
        receptive fields for neurons in V1.
        
        The input variable 'feature' is the feature or feature set to be
        expanded. If feature has only one non-zero element in all its
        groups, then the expanded feature represents the receptive field for that
        feature. If it has multiple non-zero elements, for instance, all the
        active features at one point in time, then the expanded feature
        represents the collective perception BECCA has of its environment at that
        time.

        """
        
        # Expands any active features, group by group, starting at the
        # highest-numbered group and counting down. 
        # Features from higher numbered groups always expand into lower numbered
        # groups. Counting down allows for a cascading expansion of high-level
        # features into their lower level component parts, until they are
        # completely broken down into raw sensory and basic feature inputs.
        

        
        for group_index in range(len(feature), 2, -1):

            # Checks whether there are any nonzero elements in 'feature[group_index]' 
            # that need to be expanded.
            if np.count_nonzero(feature[group_index]):

                # Finds contributions of all lower groups to the current group,
                # again counting down to allow for cascading downward projections.
                # 'parent_group_index' is a group counter variable, specific to propogating 
                # downward projections.
                for parent_group_index in range(group_index-1, -1, -1):
                    relevant_input_map_elements = (self.grouper.input_map[group_index][:,1] == parent_group_index).nonzero()
                    relevant_inputs = self.grouper.input_map[group_index][relevant_input_map_elements,0]

                    # Checks whether there are any relevant inputs to project back
                    # to group 'parent_group_index'
                    if relevant_inputs.size:

                        # Expands each feature element of group 'group_index' down to 
                        # the lower level features that it consists of.
                        # 'feature_index' is a counter variable for the features within group
                        # 'group_index'
                        for feature_index in range (feature[group_index].shape[0]):

                            # Checks whether the current feature 'feature_index' in group 'group_index' 
                            # is nonzero.
                            if feature[group_index][feature_index] != 0:

                                # Translates the feature element to its separate
                                # lower level input elements one by one.

                                # 'propogation_strength' is the amount that
                                # each element in the feature map contributes
                                # to the feature being expanded. The square
                                # root is included to offset the squaring that
                                # occurs during the upward voting process. (See
                                # agent_update_feature_map.m) 
                                propogation_strength = np.sqrt( self.feature_map.map[group_index][feature_index, relevant_input_map_elements].transpose())

                                # 'propogated_activation' is the propogation
                                # strength scaled by the activity of the
                                # high-level feature being expanded.
                                propogated_activation = propogation_strength * feature[group_index][feature_index]

                                # The lower-level feature is incremented 
                                # according to the 'propogated_activation'. The
                                # incrementing is done nonlinearly to ensure
                                # that the activity of the lower level features
                                # never exceeds 1.
                                feature[parent_group_index][relevant_inputs] = \
                                    utils.bounded_sum(feature[parent_group_index] [relevant_inputs], propogated_activation)

                #eliminates the original representation of the feature,
                #now that it is expressed in terms of lower level features
                feature[group_index] = np.zeros( feature[group_index].shape[0])
        return feature



    def step(self, sensors, primitives, reward):
        """
        A general reinforcement learning agent, modeled on human neurophysiology 
        and performance.  Takes in a time series of 
        sensory input vectors and a scalar reward and puts out a time series 
        of motor commands.  
        New features are created as necessary to adequately represent the data.
        """

        self.step_counter += 1
        logging.debug("Step: %s" % self.step_counter)

        self.sensors = sensors.copy()
        self.primitives = primitives.copy()

        # looks only at the *change* in reward as reward
        self.reward = reward

        ##############################################################
        # Feature creator
        ##############################################################

        # Breaks inputs into groups and creates new feature groups when warranted.
        # [self.grouper grouped_input group_added] = ...
        #     grouper_step( self.grouper, self.sensors, ...
        #     self.primitives, self.feature_activity)
        # TODO: add actions to sensed features
        grouped_input, group_added = self.grouper.step(self.sensors, self.primitives, self.action, self.feature_activity)
        if group_added:
            self.add_group( len(grouped_input[-1])) 


        # Interprets inputs as features and updates feature map when appropriate.
        # Assigns self.feature_activity.
        self.update_feature_map( grouped_input)

        ##############################################################
        # Reinforcement learner
        ##############################################################
        self.previous_attended_feature = copy.deepcopy(self.attended_feature)
        # Attends to a single feature. Updates self.attended_feature.
        self.attend()

        # Performs leaky integration on attended feature to get 
        # working memory.
        self.pre_previous_working_memory = copy.deepcopy(self.previous_working_memory)
        self.previous_working_memory = copy.deepcopy(self.working_memory)
        self.working_memory = self.integrate_state(self.previous_working_memory, self.attended_feature, 
                                                   self.WORKING_MEMORY_DECAY_RATE)

        # associates the reward with each transition
        # self.model.train(self.feature_activity, self.previous_working_memory, self.reward)
        self.model.train(self.feature_activity, self.pre_previous_working_memory, self.previous_attended_feature, reward)

        # Reactively chooses an action.
        # TODO: make reactive actions habit based, not reward based
        # also make reactive actions general
        # reactive_action = self.planner.select_action(self.model, self.feature_activity)

        # only acts deliberately on a fraction of the time steps
        if np.random.random_sample() > self.planner.OBSERVATION_FRACTION:
            # occasionally explores when making a deliberate action.
            # Sets self.planner.action
            if np.random.random_sample() < self.planner.EXPLORATION_FRACTION:
                self.logger.debug('EXPLORING')
                self.planner.explore()
            else:
                self.logger.debug('DELIBERATING')
                # Deliberately choose features as goals, in addition to actions.
                self.planner.deliberate(self)

        else:
            self.planner.action = np.zeros( self.planner.action.shape[0])

        # DEBUG
        # self.action = utils.bounded_sum( reactive_action, self.planner.action)
        self.action = self.planner.action

        # #debug
        # print('========================')
        # print('wm:')
        # print(self.working_memory[1].transpose())
        # print('fa:')
        # print(self.feature_activity[1].transpose())
        # print('ra:')
        # print(reactive_action.transpose())
        # print('da:')
        # print(self.planner.action.transpose())
        # print('aa:')
        # print(self.action.transpose())

        
