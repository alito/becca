
from model import Model
from planner import Planner

import numpy as np


class Actor(object):
    """ The reinforcement learner portion of the Becca agent """

    def __init__(self, num_primitives, num_actions, max_num_features):

        self.SALIENCE_NOISE = 10 ** -3 
        self.SALIENCE_WEIGHT = 0.2
        self.FATIGUE_DECAY_RATE = 10 ** -1 

        self.num_primitives = num_primitives
        self.num_actions = num_actions
        self.max_num_features = max_num_features
    
        self.model = Model(num_primitives, num_actions, self.max_num_features)
        self.planner = Planner(num_primitives, num_actions)

        self.goal = np.zeros((self.max_num_features,1))
        self.action = np.zeros((num_actions,1))
        self.salience_fatigue = np.zeros((self.max_num_features, 1))
        self.deliberately_acted = False
            
        
    def step(self, feature_activity, reward, n_features):
        
        self.feature_activity = feature_activity
        self.model.num_features = n_features
        
        """ Attend to a single feature """
        self.attended_feature = self.attend(self.deliberately_acted, self.action)

        """ Update the model """
        self.model.step(self.attended_feature, self.feature_activity, reward)

        """ Decide on an action """
        self.action, self.deliberately_acted = self.planner.step(self.model)
        
        """ debug
        Uncomment these two lines to choose a random action at each time step.
        """
        #self.action = np.zeros(self.action.size, 1);
        #self.action[np.random.randint(self.action.size), 0] = 1

        return self.action


    def attend(self, deliberately_acted, last_action=None):
        """ Select a feature from feature_activity to attend to """

        self.attended_feature = np.zeros((self.max_num_features,1))


        if (deliberately_acted and np.count_nonzero(last_action)):
            self.attended_feature[self.num_primitives: \
                                  self.num_primitives + self.num_actions,:] = last_action
           
        else:
            """ Salience is a combination of feature activity magnitude, 
            reward magnitude, goal magnitude, and a small amount of noise.
            """    
            current_feature_activity = self.feature_activity
            n_features = current_feature_activity.size
            current_goal = self.goal[:n_features,:]
            
            """ Large magnitude features are salient """
            salience = np.copy(current_feature_activity)
            
            """ Make some noise """
            noise = 1 + self.SALIENCE_NOISE / np.random.random_sample(salience.shape)
            salience *= noise
 
            """ Features associated with transitions that lead to 
            large reward are salient.
            """
            feature_salience = self.model.get_feature_salience(current_feature_activity)
            salience *= 1 + feature_salience * self.SALIENCE_WEIGHT

            """ Penalize salience for recently-attended features """
            salience *= 1 - self.salience_fatigue[:n_features,:]
                        
            """ Pick the feature with the greatest salience """
            max_salience_index = np.argmax(salience)
            
            """ Assign a 1 to the feature to be attended. Handle primitive
            and action groups according to their group numbers.
            """
            self.attended_feature[max_salience_index] = 1

            """ update fatigue """
            self.salience_fatigue[:n_features,:] += \
                (self.attended_feature[:n_features,:] - 
                self.salience_fatigue[:n_features,:]) * self.FATIGUE_DECAY_RATE
                
        return self.attended_feature

                 
    def visualize(self, save_eps=True):
        '''
        import viz_utils
        viz_utils.visualize_model(self.model, 10)
        viz_utils.force_redraw()
        '''
        return