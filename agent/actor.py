
from model import Model
from planner import Planner
import utils

import numpy as np


class Actor(object):
    """ The reinforcement learner portion of the Becca agent """

    def __init__(self, num_primitives, num_actions, max_num_features):

        self.SALIENCE_WEIGHT = 0.2

        self.num_primitives = num_primitives
        self.num_actions = num_actions
        self.max_num_features = max_num_features
    
        self.model = Model(num_primitives, num_actions, self.max_num_features)
        self.planner = Planner(num_primitives, num_actions, self.max_num_features)

        self.action = np.zeros((num_actions,1))
            
        
    def step(self, feature_activity, reward, n_features):
        
        self.feature_activity = feature_activity
        self.model.num_features = n_features
        
        """ Attend to a single feature """
        #self.attended_feature = self.attend()

        """ Update the model """
        #self.model.step(self.attended_feature, self.feature_activity, reward)
        self.model.step(self.feature_activity, reward)

        """ Decide on an action """
        #self.action = self.planner.step(self.model, self.attended_feature, n_features)
        self.action = self.planner.step(self.model, n_features)
        
        """ debug
        Uncomment these two lines to choose a random action at each time step.
        """
        #self.action = np.zeros(self.action.shape);
        #self.action[np.random.randint(self.action.size), 0] = 1

        #print 'final action', self.action.ravel() 
        return self.action


    def attend(self):

        self.attended_feature = np.zeros((self.max_num_features,1))

        """ Salience is a combination of feature activity magnitude, 
        reward magnitude, goal magnitude, and a small amount of noise.
        """    
        salience = np.copy(self.feature_activity) + utils.EPSILON
        
        """ Jitter the salience values """
        salience += utils.EPSILON * np.random.random_sample(salience.shape)
        #self.SALIENCE_WEIGHT = 10
        salience *= 1 + self.planner.goal #* self.SALIENCE_WEIGHT
        #salience *= 1 - model.prediction
        cumulative_salience = np.cumsum(salience,axis=0) / np.sum(salience, axis=0)
        attended_feature_index = np.nonzero(np.random.random_sample() < cumulative_salience)[0][0]
        
        '''print 'self.feature_activity', self.feature_activity[np.nonzero(self.feature_activity)[0],:].ravel(), \
                                np.nonzero(self.feature_activity)[0].ravel()
        print 'self.planner.goal', self.planner.goal[np.nonzero(self.planner.goal)[0],:].ravel(), \
                                np.nonzero(self.planner.goal)[0].ravel()
        print 'attended_feature_index', attended_feature_index
        '''
        self.attended_feature[attended_feature_index] = 1.
        
        return self.attended_feature

                 
    def visualize(self, save_eps=True):
        
        import viz_utils
        viz_utils.visualize_model(self.model, self.num_primitives, self.num_actions, 10)
        viz_utils.force_redraw()
        
        return