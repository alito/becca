
from model import Model
from planner import Planner
from state import State
import utils
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
        
        """ These constants are used to adaptively track the mean 
        and distribution of reward, so that typical rewards can be mapped 
        onto (-1,1) in a meaningful way.
        """
        self.reward_average = 0.
        self.reward_deviation = 1.
        self.REWARD_AVERAGE_DECAY_RATE = 3 * 10. ** -3
        self.REWARD_DEVIATION_DECAY_RATE = 3 * 10. ** -3
        
        """ Reward offset is a 'life is good' constant. It introduces
        a baseline of contentment to the agent.
        """
        self.REWARD_OFFSET = 0.1
        

    def step(self, feature_activity, raw_reward, n_features):
        
        self.feature_activity = feature_activity
        self.model.n_features = n_features
        
        #debug
        reward = self.process_reward(raw_reward)
        #reward = raw_reward
        
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
        
        debug = False
        if np.random.random_sample() < 0.01:
            debug = True
        if debug:
            print 'before', 'self.reward_average', self.reward_average,  \
                    'self.reward_deviation', self.reward_deviation, \
                    'raw_reward', raw_reward
                    
        """ Map raw reward onto subjective reward """
        self.reward_average = self.reward_average * \
                (1. - self.REWARD_AVERAGE_DECAY_RATE) \
                + self.REWARD_AVERAGE_DECAY_RATE * raw_reward
        
        self.reward_deviation = self.reward_deviation * \
                (1. - self.REWARD_DEVIATION_DECAY_RATE) \
                + self.REWARD_DEVIATION_DECAY_RATE * \
                np.abs(raw_reward - self.reward_average)
        
        reward = utils.map_inf_to_one((raw_reward - self.reward_average) / \
                                self.reward_deviation) + self.REWARD_OFFSET
         
        if debug:                       
            print 'after', 'self.reward_average', self.reward_average,  \
                    'self.reward_deviation', self.reward_deviation, \
                    'reward', reward

        return reward
    
    
    def attend(self, deliberately_acted, last_action=None):
        """ Selects a feature from feature_activity to attend to """

        self.attended_feature = self.feature_activity.zeros_like()

        if deliberately_acted and np.count_nonzero(last_action):
            self.attended_feature.set_actions(last_action)
            
        else:
            """ Salience is a combination of feature activity magnitude, 
            reward magnitude, goal magnitude, and a small amount of noise.
            """    
            current_feature_activity = self.feature_activity.features
            n_features = current_feature_activity.size
            current_goal = self.goal.features[n_features,:]
            
            #context_feature_values = self.model.context.features[
            #           :n_features,:self.model.n_transitions] * \
            #    self.model.reward_value[
            #           :self.model.n_transitions,:].transpose()
                       
            '''context_feature_totals = self.model.context.features[
                       :n_features,:self.model.n_transitions] * \
                abs(self.model.reward_value[
                       :self.model.n_transitions,:].transpose()) * \
                self.model.count[
                       :self.model.n_transitions,:].transpose()
                       
            context_feature_counts = self.model.context.features[
                       :n_features,:self.model.n_transitions] * 
                self.model.count[
                       :self.model.n_transitions,:].transpose()
                       
            context_feature_values = context_feature_totals / \
                                    context_feature_counts
            '''                       
            #context_feature_value_uncertainty = self.model.context.features[
            #           :n_features,:self.model.n_transitions] * \
            #    self.model.reward_uncertainty[
            #           :self.model.n_transitions,:].transpose()
                       
            '''print '---'
            print self.model.context.features[
                       :n_features,:self.model.n_transitions] .transpose()
            print self.model.reward_value[
                       :self.model.n_transitions,:].transpose()
            print (self.model.context.features[
                       :n_features,:self.model.n_transitions] * \
                self.model.reward_value[
                       :self.model.n_transitions,:].transpose()).transpose()
            '''
            #feature_values = utils.bounded_array_sum(
            #        context_feature_values, dim=1) 
            #feature_values = np.sum(context_feature_values, axis=1)
            
            #if context_feature_values.size > 0:
            #    feature_values = np.amax(context_feature_values, axis=1)
            #else:
            #    feature_values =    context_feature_values
            '''
            if context_feature_value_uncertainty.size > 0:
                feature_values = np.amax(1-context_feature_value_uncertainty, axis=1)
            else:
                feature_values =    context_feature_value_uncertainty
            '''                 
            
            #debug
            #if np.random.random_sample() < 0.01:
            #    print context_feature_values
            #    print feature_values
            
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