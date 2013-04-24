import numpy as np

from map import Map
from model import Model

class Cog(object):
    """ 
    The building block of which levels are composed

    Cogs are named for their similarity to clockwork cogwheels.
    They are simple and do the same task over and over, but by
    virtue of how they are connected to their fellows, they 
    collectively bring about interesting behavior.  

    Each cog contains two important parts, a model and a map.
    During upward processing, feature inputs are used to train
    the model and the map. During downward processing, 
    the model and map use the goal inputs from the next level to
    create lower-level goals. 
    """
    def __init__(self, max_feature_inputs, max_feature_outputs, 
                 name='anonymous'):
        """ Initialize the cogs with a pre-determined maximum size """
        self.name = name
        self.max_feature_inputs = max_feature_inputs
        self.max_feature_outputs = max_feature_outputs
        self.model = Model(max_feature_inputs, name=name)        
        if max_feature_outputs > 0:
            self.map = Map(max_feature_inputs **2, max_feature_outputs, 
                           name=name)

    def step_up(self, feature_input, reward):
        """ Let feature_input percolate upward through the model and map """
        transition_activities = self.model.update(feature_input, reward) 
        self.reaction= self.model.get_reaction()
        self.surprise = self.model.get_surprise()
        feature_output = self.map.update(transition_activities)
        return feature_output

    def step_down(self, goal_input):
        """ Let goal_input percolate downward through the map and model """
        transition_goals = self.map.get_transition_goals(goal_input) 
        goal_vote = self.model.deliberate(transition_goals)     
        self.goal_output = self.model.get_goal()
        return goal_vote

    def get_projection(self, feature_index):
        """ Project a feature down through the map and model """
        map_projection = self.map.get_projection(feature_index)
        model_projection = self.model.get_projection(map_projection)
        return model_projection
         
    def fraction_filled(self):
        """ How full is the input set for this cog? """
        return (float(self.model.num_feature_inputs) / 
                float(self.max_feature_inputs))
        
    def display(self):
        """ Show the internal state of the model and map """
        self.model.visualize()
        if self.max_feature_outputs > 0:
            self.map.visualize()
        return
