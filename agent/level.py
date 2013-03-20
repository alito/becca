
from cog import Cog
import utils

import numpy as np

class Level(object):
    def __init__(self, cogs, name='anonymous'):
        self.cogs = cogs
        self.num_cogs = len(self.cogs)
        self.name = name
        
        
    def step_up(self, features, reward):
        last_feature = 0
        hi_feature_activities = np.zeros((0,1))
        for cog in self.cogs:
            cog_hi_feature_activities = \
                        cog.step_up(features[last_feature:last_feature + cog.max_num_features,:], reward)
            hi_feature_activities= np.vstack((hi_feature_activities, cog_hi_feature_activities))
            last_feature += cog.max_num_features
        return hi_feature_activities


    def step_down(self, hi_goal=np.zeros((0,1))):
        last_goal = 0
        goal = np.zeros((0,1))
        for cog in self.cogs:
            cog_goal = cog.step_down(hi_goal[last_goal:last_goal + cog.max_num_hi_features,:])
            goal= np.vstack((goal, cog_goal))
            last_goal += cog.max_num_hi_features
        return goal 


    def get_projections(self):
        projections = []
        for cog in self.cogs:
            projections.append(cog.get_projections())
        return projections
        
        
    def display(self):
        for cog in self.cogs:
            cog.display()
        return
