
import copy
import numpy as np
from model import Model
from planner import Planner
from state import State
import viz_utils

class Learner(object):
    """ The reinforcement learner portion of the Becca agent """

    def __init__(self, num_primitives, num_actions):

        """ Sensors are irrelevant in the learner """
        num_sensors = 0
        
        self.SALIENCE_NOISE = 0.001        
        
        self.model = Model(num_primitives, num_actions)
        self.planner = Planner(num_actions)

        self.goal = State(num_sensors, num_primitives, num_actions)
        self.action = np.zeros((num_actions,1))
        self.deliberately_acted = False
        

    def step(self, feature_activity, reward):
        
        self.maintain_state_size(feature_activity)
        
        """ Attend to a single feature """
        self.attended_feature = self.attend(feature_activity, 
                                            self.deliberately_acted, 
                                            self.action)
        
        """ Update the model """
        self.model.step( self.attended_feature, 
                         feature_activity, reward)

        """ Decide on an action """
        self.action, self.deliberately_acted = self.planner.step(self.model)
        
        """ debug: choose a random action """
        #self.action = np.zeros(self.goal.action.size, 1);
        #self.action[np.random.randint(self.goal.action.size), 0] = 1

        return self.action


    def maintain_state_size(self, feature_activity):
        """ Checks whether feature_activity is larger than any of 
        the learner's state variables, and grows them as appropriate
        by adding states and features.
        """
        n_learner_feature_groups = self.model.n_feature_groups()
        
        if n_learner_feature_groups < feature_activity.n_feature_groups():
            n_features = feature_activity.features[-1].size                                           
            self.goal.add_group(n_features)
            self.model.add_group(n_features)
        return
    
    
    def attend(self, feature_activity, deliberately_acted, last_action=None):
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
                
        """ Calculate salience for action """
        (max_salience_value, max_salience_group, max_salience_index) = \
            self.calculate_salience(salience.action, 
                                   feature_activity.action, 
                                   self.goal.action, 
                                   max_salience_value, max_salience_group, 
                                   max_salience_index,
                                   group_indx=-1)
                
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
            self.attended_feature.action[max_salience_index] = 1

        else:
            self.attended_feature.features[max_salience_group] \
                                          [max_salience_index] = 1

        if deliberately_acted:
            if np.count_nonzero(last_action):
                self.attended_feature = feature_activity.zeros_like()
                self.attended_feature.action = last_action
            
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
                 
        return max_salience_val, max_salience_grp, max_salience_indx

                   
    def visualize(self, save_eps=True):
        viz_utils.visualize_model(self.model, 10)
        viz_utils.force_redraw()
        
        return