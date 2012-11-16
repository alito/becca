
from model import Model
from planner import Planner
from state import State
import viz_utils

import numpy as np


class Actor(object):
    """ The reinforcement learner portion of the Becca agent """

    def __init__(self, num_primitives, num_actions, max_num_features):

        self.SALIENCE_NOISE = 0.001        
        self.MAX_NUM_FEATURES = max_num_features
    
        self.model = Model(num_primitives, num_actions, self.MAX_NUM_FEATURES)
        self.planner = Planner(num_actions)

        self.goal = State(num_primitives, num_actions, self.MAX_NUM_FEATURES)
        self.action = np.zeros((num_actions,1))
        self.deliberately_acted = False
        

    def step(self, feature_activity, reward, n_features):
        
        self.feature_activity = feature_activity
        self.model.n_features = n_features
        
        """ Attend to a single feature """
        self.attended_feature = self.attend(self.deliberately_acted, 
                                            self.action)
        
        """ Update the model """
        self.model.step(self.attended_feature, self.feature_activity, reward)

        """ Decide on an action """
        self.action, self.deliberately_acted = self.planner.step(self.model)
        
        """ debug
        Uncomment these two lines to choose a random action at each time step.
        """
        #self.action = np.zeros(self.goal.action.size, 1);
        #self.action[np.random.randint(self.goal.action.size), 0] = 1

        return self.action


    def attend(self, deliberately_acted, last_action=None):
        """ Selects a feature from feature_activity to attend to """

        self.attended_feature = self.feature_activity.zeros_like()

        if deliberately_acted and np.count_nonzero(last_action):
            self.attended_feature.set_actions(last_action)
            
        else:
            """ Salience is a combination of feature activity magnitude, 
            goal magnitude, and a small amount of noise.
            """    
            current_feature_activity = self.feature_activity.features
            n_features = current_feature_activity.size
            current_goal = self.goal.features[:n_features,:]
            salience = np.zeros_like(current_feature_activity)
            
            salience = self.SALIENCE_NOISE * \
                        np.random.random_sample(salience.shape)
             
            salience += current_feature_activity * (1 + current_goal)
                        
            """ Pick the feature with the greatest salience """
            max_salience_index = np.argmax(salience)
    
            """ Assign a 1 to the feature to be attended. Handle primitive
            and action groups according to their group numbers.
            """
            self.attended_feature.features[max_salience_index] = 1

        return self.attended_feature

                 
    def visualize(self, save_eps=True):
        #viz_utils.visualize_model(self.model, 10)
        #viz_utils.force_redraw()
        
        return