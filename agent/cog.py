
from model import Model
from map import Map
from utils import *
import numpy as np

class Cog(object):

    def __init__(self, max_num_features, max_num_hi_features, name='anonymous'):
        self.REPORTING_PERIOD = 10 ** 3
        self.name = name
        self.max_num_features = max_num_features
        self.max_num_hi_features = max_num_hi_features
        self.model = Model(max_num_features, name=name)        
        if max_num_hi_features > 0:
            self.map = Map(max_num_features **2, max_num_hi_features, name=name)

        
    def step_up(self, features, reward):
        """ Pad the incoming features array out to its full size if necessary """
        #features = np.vstack((features, np.zeros((self.max_num_features - features.size, 1))))
        transition_activities = self.model.update(features, reward)
        
        if self.max_num_hi_features > 0:
            hi_feature_activities = self.map.update(transition_activities)
        
            """ Pad the outgoing features array out to its full size if necessary """
            hi_feature_activities = np.vstack((hi_feature_activities, 
                                   np.zeros((self.max_num_hi_features - hi_feature_activities.size, 1))))
        else:
            hi_feature_activities = np.zeros((0,1))
        return hi_feature_activities

        
    def step_down(self, hi_goal):
        """ Pad the incoming features array out to its full size if necessary """
        if self.max_num_hi_features > 0:
            transition_goals = self.map.get_transition_goals(hi_goal) 
        else:
            transition_goals = np.zeros((self.max_num_features, 1))            
        goal = self.model.deliberate(transition_goals)     
        return goal

    def get_projections(self):
        map_projections = self.map.get_projections()
        return self.model.get_projections(map_projections)
        
    def display(self):
        self.model.visualize()
        if self.max_num_hi_features > 0:
            self.map.visualize()
        return
