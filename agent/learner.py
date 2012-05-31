
import copy
import numpy as np
from model import Model
from planner import Planner
from state import State
import viz_utils

class Learner(object):
    """ The reinforcement learner portion of the Becca agent """

    def __init__(self, num_real_primitives, num_actions):

        """ Sensors are irrelevant in the learner """
        num_sensors = 0
        
        self.SALIENCE_NOISE = 0.001        
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
        
        self.model = Model(num_real_primitives, num_actions)
        self.planner = Planner(num_actions)

        self.attended_feature = State(num_sensors, num_real_primitives, num_actions)
        self.goal = State(num_sensors, num_real_primitives, num_actions)
        self.previous_working_memory = State(num_sensors, num_real_primitives, 
                                             num_actions)
        self.working_memory = State(num_sensors, num_real_primitives, num_actions)


    def step(self, feature_activity, reward):
        
        self.grow_states(feature_activity)
        
        self.previous_attended_feature = copy.deepcopy(self.attended_feature)
        
        """ Attend to a single feature """
        self.attended_feature = self.attend(feature_activity)
        
        """ Perform leaky integration on attended feature to get 
        working memory.
        """        
        self.pre_previous_working_memory = \
                    copy.deepcopy(self.previous_working_memory)
        self.previous_working_memory = copy.deepcopy(self.working_memory)
        self.working_memory = self.previous_working_memory.integrate_state(
                                           self.attended_feature, 
                                           self.WORKING_MEMORY_DECAY_RATE)
        
        """ Associate the reward with each transition """
        self.model.step(self.pre_previous_working_memory, 
                         self.previous_attended_feature, 
                         feature_activity, reward)

        """ Decide on an action """
        self.actions, deliberately_acted = \
                        self.planner.step(self.model, self.working_memory)
        
        """ If a deliberate action was made on this timestep,
        force the agent to attend to it. This ensures that 
        exploratory actions will be attended.
        """  
        if deliberately_acted:
            self.attended_feature = self.attended_feature.zeros_like()
            self.attended_feature.actions = self.actions
            self.working_memory = self.previous_working_memory.integrate_state(
                                               self.attended_feature, 
                                               self.WORKING_MEMORY_DECAY_RATE)
                
            
        """ debug: choose a random action """
        #self.actions = np.zeros(self.goal.actions.size);
        #self.actions[np.random.randint(self.goal.actions.size)] = 1

        return self.actions


    def grow_states(self, feature_activity):
        """ Checks whether feature_activity is larger than any of 
        the learner's state variables, and grows them as appropriate
        by adding states and features.
        """
        n_learner_feature_groups = self.goal.n_feature_groups()
        
        for group_index in range(feature_activity.n_feature_groups()):
            
            """ Add the group if necessary """
            if group_index >= n_learner_feature_groups:
                self.add_group()

            """ Add as many features as necessary """
            n_features = feature_activity.n_features_in_group(group_index)
            n_learner_features = self.goal.n_features_in_group(group_index)
            for feature_count in range(n_features - n_learner_features):
                self.add_feature(group_index)

        return
    
    
    def attend(self, feature_activity):
        """ Selects a feature from feature_activity to attend to """

        max_salience_value = 0
        max_salience_group = 0
        max_salience_index = 0

        """ Salience is a combination of feature activity magnitude, 
        goal magnitude, and a small amount of noise.
        """    
        salience = feature_activity.zeros_like()
        self.attended_feature = feature_activity.zeros_like()
        
        """ Calculate salience for primitives """
        (max_salience_value, max_salience_group, max_salience_index) = \
            self.calculate_salience(salience.primitives, 
                                   feature_activity.primitives, 
                                   self.goal.primitives, 
                                   max_salience_value, max_salience_group, 
                                   max_salience_index, group_indx=-2)
                
        """ Calculate salience for actions """
        (max_salience_value, max_salience_group, max_salience_index) = \
            self.calculate_salience(salience.actions, 
                                   feature_activity.actions, 
                                   self.goal.actions, 
                                   max_salience_value, max_salience_group, 
                                   max_salience_index,
                                   group_indx=-1)
        # group_indx=-1, deliberate=self.planner.deliberately_acted)
                
        """ Calculate salience for feature groups """
        for group_index in range(feature_activity.n_feature_groups()):
            
            if feature_activity.n_features_in_group(group_index) > 0:
                (max_salience_value, max_salience_group, max_salience_index) =\
                    self.calculate_salience(salience.features[group_index], 
                                   feature_activity.features[group_index], 
                                   self.goal.features[group_index], 
                                   max_salience_value, max_salience_group, 
                                   max_salience_index, group_indx=group_index)

        """ Assign a 1 to the feature to be attended. Handle primitive
        and action groups according to their group numbers.
        """
        if max_salience_group == -2:
            self.attended_feature.primitives[max_salience_index] = 1

        elif max_salience_group == -1:
            self.attended_feature.actions[max_salience_index] = 1

        else:
            self.attended_feature.features[max_salience_group] \
                                          [max_salience_index] = 1

        # debug
        '''
        if np.random.random_sample(1) < 0.01:
            print "attention report "
            viz_utils.visualize_state(feature_activity, label='feature_activity')
            viz_utils.visualize_state(self.attended_feature, label='attended_feature')
            import matplotlib.pyplot
            matplotlib.pyplot.show()
        '''
            
        return self.attended_feature


    def calculate_salience(self, salience, feature_activity, goal,
                           max_salience_val, max_salience_grp, 
                           max_salience_indx, group_indx, deliberate=False):
    
        salience = self.SALIENCE_NOISE * np.random.random_sample(salience.shape)
        salience += feature_activity * (1 + goal)
                    
        """ Pick the feature with the greatest salience """
        max_value = np.max(salience)
        max_index = np.argmax(salience)
        
        if max_value >= max_salience_val:
            max_salience_val = max_value
            max_salience_grp = group_indx
            max_salience_indx = max_index
            
        """ If a deliberate action was made on the previous timestep,
        force the agent to attend to it. This ensures that 
        exploratory actions will be attended.
        """  
        '''                  
        if deliberate:
            if np.count_nonzero(feature_activity != 0.0):
                deliberate_action_index = feature_activity.nonzero()[0]
                max_salience_val = 10
                max_salience_grp = -1
                max_salience_indx = deliberate_action_index
        '''
                
        return max_salience_val, max_salience_grp, max_salience_indx
    

    def add_group(self):
                
        self.working_memory.add_group()
        self.previous_working_memory.add_group()
        self.attended_feature.add_group()
        self.goal.add_group()
        self.model.add_group()
        return


    def add_feature(self, nth_group):
        
        self.working_memory.add_feature(nth_group)
        self.previous_working_memory.add_feature(nth_group)
        self.attended_feature.add_feature(nth_group)
        self.goal.add_feature(nth_group)
        self.model.add_feature(nth_group)
        return
    
                   
    def visualize(self, save_eps=True):
        #viz_utils.visualize_model(self.model, 10)
        viz_utils.force_redraw()
        
        return