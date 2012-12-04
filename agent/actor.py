
import copy
from model import Model
from planner import Planner
from state import State
import utils
import viz_utils

import numpy as np


class Actor(object):
    """ The reinforcement learner portion of the Becca agent """

    def __init__(self, num_primitives, num_actions, max_num_features):

        self.SALIENCE_NOISE = 10 ** -3 
        self.SALIENCE_WEIGHT = 0.2
        self.FATIGUE_DECAY_RATE = 10 ** -1 
        self.MAX_NUM_FEATURES = max_num_features
    
        self.model = Model(num_primitives, num_actions, self.MAX_NUM_FEATURES)
        self.planner = Planner(num_actions)

        self.goal = State(num_primitives, num_actions, self.MAX_NUM_FEATURES)
        self.action = np.zeros((num_actions,1))
        self.salience_fatigue = np.zeros((self.MAX_NUM_FEATURES, 1))
        self.deliberately_acted = False
        
        """ These constants are used to adaptively track the mean 
        and distribution of reward, so that typical rewards can be mapped 
        onto (-1,1) in a meaningful way.
        """
        self.reward_average = 0.
        self.reward_deviation = 0.25
        self.REWARD_AVERAGE_DECAY_RATE = 10. ** -1
        self.REWARD_DEVIATION_DECAY_RATE = self.REWARD_AVERAGE_DECAY_RATE * \
                                            10. ** -2
           
        self.BIGGEST_REWARD = utils.EPSILON
        
        
    def step(self, feature_activity, raw_reward, n_features):
        
        self.feature_activity = feature_activity
        self.model.n_features = n_features
        
        #debug
        #reward = self.process_reward(raw_reward)
        #reward = raw_reward / 2
        #reward = utils.map_inf_to_one(raw_reward)
        
        if np.abs(raw_reward) > self.BIGGEST_REWARD:
            self.BIGGEST_REWARD = np.abs(raw_reward)
        
        reward = raw_reward / self.BIGGEST_REWARD
        
        
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


    def process_reward(self, raw_reward):
        
        """ Map raw reward onto subjective reward """
        self.reward_average = self.reward_average * \
                (1. - self.REWARD_AVERAGE_DECAY_RATE) \
                + self.REWARD_AVERAGE_DECAY_RATE * raw_reward
        
        self.reward_deviation = self.reward_deviation * \
                (1. - self.REWARD_DEVIATION_DECAY_RATE) \
                + self.REWARD_DEVIATION_DECAY_RATE * \
                np.abs(raw_reward - self.reward_average)
        
        reward = utils.map_inf_to_one((raw_reward - self.reward_average) / \
                                self.reward_deviation)
         
        #debug
        #print 'difference', reward - raw_reward, ' reward', reward,' raw reward', raw_reward
        
        return reward
    
    
    def attend(self, deliberately_acted, last_action=None):
        """ Selects a feature from feature_activity to attend to """

        self.attended_feature = self.feature_activity.zeros_like()
        self.attended_feature.features = np.zeros((self.MAX_NUM_FEATURES,1))

        if deliberately_acted and np.count_nonzero(last_action):
            self.attended_feature.set_actions(last_action)
            
        else:
            """ Salience is a combination of feature activity magnitude, 
            reward magnitude, goal magnitude, and a small amount of noise.
            """    
            current_feature_activity = self.feature_activity.features
            n_features = current_feature_activity.size
            current_goal = self.goal.features[:n_features,:]
            
            debug = False
            if np.random.random_sample() < 0.0:
                debug = True
            
            """ Large magnitude features are salient """
            salience = copy.deepcopy(current_feature_activity)
            
            if debug:
                print 'feature_activity', current_feature_activity.ravel()
            
            """ Make some noise """
            #salience = self.SALIENCE_NOISE * \
            #            np.random.random_sample(salience.shape)
            noise = 1 + self.SALIENCE_NOISE / \
                        np.random.random_sample(salience.shape)
            salience *= noise
            if debug:
                print 'noise', noise.ravel()
                print 'salience w noise', salience.ravel()
 
            """ Features associated with transitions that lead to 
            large reward are salient.
            """
            feature_salience = self.model.get_feature_salience(current_feature_activity) 
            
            salience *= 1 + feature_salience * self.SALIENCE_WEIGHT
            
            if debug:
                print 'feature salience', 1 + feature_salience.ravel()
                print 'salience w features', salience.ravel()

            """ Penalize salience for recently-attended features """
            salience *= 1 - self.salience_fatigue[:n_features,:]
                        
            if debug:
                print 'fatigue', 1 - self.salience_fatigue[:n_features,:].ravel()
                print 'salience w fatigue', salience.ravel()
            
            """ Pick the feature with the greatest salience """
            max_salience_index = np.argmax(salience)
            
            if debug:
                print 'Attended feauture: ', max_salience_index
            
            """ Assign a 1 to the feature to be attended. Handle primitive
            and action groups according to their group numbers.
            """
            self.attended_feature.features[max_salience_index] = 1

            """ update fatigue """
            self.salience_fatigue[:n_features,:] += \
                (self.attended_feature.features[:n_features,:] - 
                self.salience_fatigue[:n_features,:]) * self.FATIGUE_DECAY_RATE
                
        return self.attended_feature

                 
    def visualize(self, save_eps=True):
        #viz_utils.visualize_model(self.model, 10)
        #viz_utils.force_redraw()
        
        return