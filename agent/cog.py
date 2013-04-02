
from model import Model
from map import Map
from utils import *
import numpy as np

class Cog(object):

    def __init__(self, max_feature_inputs, max_feature_outputs, name='anonymous'):
        self.REPORTING_PERIOD = 10 ** 3
        self.name = name
        self.max_feature_inputs = max_feature_inputs
        self.max_feature_outputs = max_feature_outputs
        self.model = Model(max_feature_inputs, name=name)        
        if max_feature_outputs > 0:
            self.map = Map(max_feature_inputs **2, max_feature_outputs, name=name)

    def step_up(self, feature_input, reward):
        transition_activities = self.model.update(feature_input, reward)        
        feature_output = self.map.update(transition_activities)
        return feature_output

    def step_down(self, goal_input):
        transition_goals = self.map.get_transition_goals(goal_input) 
        goal_output = self.model.deliberate(transition_goals)     
        return goal_output

    def compare_prediction(self, feature_inputs):
        return self.model.compare_prediction(feature_inputs)

    def get_projection(self, feature_index):
        map_projection = self.map.get_projection(feature_index)
        return self.model.get_projection(map_projection)

    def filled(self):
        return float(self.model.num_feature_inputs) / float(self.max_feature_inputs)
        
    def display(self):
        self.model.visualize()
        if self.max_feature_outputs > 0:
            self.map.visualize()
        return
