
import copy
import numpy as np
#import matplotlib.pyplot as plt

from agent_stub import AgentStub
from feature_map import FeatureMap
from grouper import Grouper
from model import Model
from planner import Planner
from state import State
import utils

class Agent(AgentStub):
    """ A general reinforcement learning agent, modeled on observations and 
    theories of human performance. It takes in a time series of 
    sensory input vectors and a scalar reward and puts out a time series 
    of motor commands. New features are created as necessary to adequately 
    represent the data.
    """

    def __init__(self, agent_name, num_sensors, num_primitives, num_actions, 
                 max_number_features=1000):
        """ agent_name is a string label for the agent
            num_sensors, num_primitives, and num_actions are the number of 
        sensor, primitive feature, and output channels, respectively, that
        the agent expects in its interaction with the world 
            max_num_features is an upper limit on the number of 
        features the agent can produce. 
        """

        #super(Agent, self).__init__(agent_name, num_sensors, 
        #                            num_primitives, num_actions, 
        #                            max_number_features)
        
        self.pickle_filename = agent_name + "_agent.pickle"
        
        self.REPORTING_PERIOD = 10 ** 3
        self.BACKUP_PERIOD = 10 ** 4

        self.num_sensors = num_sensors
        self.num_primitives = num_primitives
        self.num_actions = num_actions

        self.reward = 0
        self.actions = np.zeros(self.num_actions)
        
        self.timestep = 0
        self.graphing = True
        
        self.cumulative_reward = 0
        self.reward_history = []
        self.reward_steps = []
        
        self.SALIENCE_NOISE = 0.1        
        self.GOAL_DECAY_RATE = 0.05   # real, 0 < x <= 1
        self.STEP_DISCOUNT = 0.5      # real, 0 < x <= 1
        
        """ Rates at which the feature activity and working memory decay.
        Setting these equal to 1 is the equivalent of making the Markovian 
        assumption about the task--that all relevant information is captured 
        adequately in the current state.
        Also check out self.grouper.INPUT_DECAY_RATE, set in 
        the grouper constructor
        """
        self.WORKING_MEMORY_DECAY_RATE = 0.4      # real, 0 < x <= 1
        
        self.grouper = Grouper( num_sensors, num_actions, num_primitives, 
                                max_number_features, graphs=self.graphing)
        self.feature_map = FeatureMap(num_sensors, num_primitives, num_actions)        
        self.model = Model( num_primitives, num_actions, graphs=self.graphing)
        self.planner = Planner(num_actions)

        self.NEW_FEATURE_MARGIN = 0.3
        self.NEW_FEATURE_MIN_SIZE = 0.2
        
        self.attended_feature = State(num_sensors, num_primitives, num_actions)
        self.feature_activity = State(num_sensors, num_primitives, num_actions)
        self.goal = State(num_sensors, num_primitives, num_actions)
        self.previous_working_memory = State(num_sensors, num_primitives, 
                                             num_actions)
        self.working_memory = State(num_sensors, num_primitives, num_actions)
               
               
    def add_group(self, group_length):
                
        self.feature_activity.add_group()
        self.working_memory.add_group()
        self.previous_working_memory.add_group()
        self.attended_feature.add_group()
        self.goal.add_group()

        self.feature_map.add_group(group_length)
        self.grouper.add_group()
        self.model.add_group()

        print("Adding group " + str(len(self.goal.features)))


    def add_feature(self, new_feature, nth_group, feature_vote):
        
        feature_vote.add_feature(nth_group)
        self.feature_activity.add_feature(nth_group)
        self.working_memory.add_feature(nth_group)
        self.previous_working_memory.add_feature(nth_group)
        self.attended_feature.add_feature(nth_group)
        self.goal.add_feature(nth_group)

        self.grouper.add_feature(nth_group)
        self.feature_map.add_feature(nth_group, new_feature)
        self.model.add_feature(nth_group)

        print("Adding feature to group %s" % nth_group)        

        return feature_vote


    def attend(self):
        """ Selects a feature from feature_activity to attend to.
        """

        max_salience_value = 0
        max_salience_group = 0
        max_salience_index = 0

        """ Salience is a combination of feature activity magnitude, 
        goal magnitude, and a small amount of noise.
        """

        salience = utils.AutomaticList()
        for group_index in range(len(self.feature_activity)):
            count = len( self.feature_activity[group_index])
            self.attended_feature[group_index] = np.zeros(count)

            if count > 0:
                # no point in doing this if the feature is empty
                salience[group_index] = self.SALIENCE_NOISE * np.random.random_sample(count)                
                salience[group_index] += self.feature_activity[group_index] * (1 + self.goal[group_index])

                # Picks the feature with the greatest salience.
                max_value = np.max(salience[group_index])
                max_index = np.argmax(salience[group_index])


                if max_value >= max_salience_value:
                    max_salience_value = max_value
                    max_salience_group = group_index
                    max_salience_index = max_index


            if self.planner.act:
                # TODO: workaround for numpy versions earlier than 1.6
                if np.count_nonzero(self.planner.action != 0.0):
                    deliberate_action_index = self.planner.action.nonzero()[0]

                    #ensures that exploratory actions will be attended
                    max_salience_value = 10
                    max_salience_group = 2
                    max_salience_index = deliberate_action_index

                #self.planner.explore = False  # this doesn't seem to be used anywhere on Becca_m


        #print('attended--group: %s, feature: %s, value: %s' % \
        #              (max_salience_group, max_salience_index, max_salience_value))


        self.attended_feature[max_salience_group][max_salience_index] = 1


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
        represents the collective perception Becca has of its environment at that
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
            # TODO: workaround for numpy versions earlier than 1.6
            if np.count_nonzero(feature[group_index]):

                # Finds contributions of all lower groups to the current group,
                # again counting down to allow for cascading downward projections.
                # 'parent_group_index' is a group counter variable, specific to propogating 
                # downward projections.
                for parent_group_index in range(group_index-1, -1, -1):
                    relevant_input_map_elements = (self.grouper.input_map[group_index][:,1] == parent_group_index).nonzero()
                    print 'relevant_input', relevant_input_map_elements
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

                                # 'propagation_strength' is the amount that
                                # each element in the feature map contributes
                                # to the feature being expanded. The square
                                # root is included to offset the squaring that
                                # occurs during the upward voting process. (See
                                # agent_update_feature_map.m) 
                                propagation_strength = np.sqrt( self.feature_map.map[group_index][feature_index, relevant_input_map_elements].transpose())

                                # 'propagated_activation' is the propagation
                                # strength scaled by the activity of the
                                # high-level feature being expanded.
                                propagated_activation = propagation_strength * feature[group_index][feature_index]

                                # The lower-level feature is incremented 
                                # according to the 'propagated_activation'. The
                                # incrementing is done nonlinearly to ensure
                                # that the activity of the lower level features
                                # never exceeds 1.
                                feature[parent_group_index][relevant_inputs] = \
                                    utils.bounded_sum(feature[parent_group_index] [relevant_inputs], propagated_activation)

                #eliminates the original representation of the feature,
                #now that it is expressed in terms of lower level features
                feature[group_index] = np.zeros( feature[group_index].shape[0])
        return feature



    def step(self, sensors, primitives, reward):
        """ Advances the agent's operation by one time step """
        
        self.timestep += 1

        self.sensors = sensors.copy()
        self.primitives = primitives.copy()
        self.reward = reward

        """
        Feature creator
        ======================================================
        """
        
        """ Breaks inputs into groups and creates new feature 
        groups when warranted.
        """
        (grouped_input, group_added) = self.grouper.step(self.sensors, 
                                                         self.primitives, 
                                                         self.action, 
                                                         self.feature_activity)
        if group_added:
            self.add_group( len(grouped_input[-1])) 
        

        """ Interprets inputs as features and updates feature map 
        when appropriate. Assigns self.feature_activity.
        """
        """
        self.feature_activity = self.update_feature_map( grouped_input)

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

        # decide on an action
        self.actions = self.planner.step(self)
        """
        self.log()
        
        self.actions = np.zeros(self.num_actions);
        self.actions[np.random.randint(self.num_actions)] = 1
        
        return self.actions
    

    def display(self):
        #print self.timestep
        if (self.timestep % self.REPORT_PERIOD) == 0:
            print 'step', self.timestep
            print 'grouper.last', self.grouper.last_entry
            #self.model.display_n_best(1)
            #utils.force_redraw()
