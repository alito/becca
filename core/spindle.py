""" the Spindle class """
import matplotlib.pyplot as plt
import numpy as np
import tools

class Spindle(object):
    """ 
    Pick one feature per timestep to pass on to the mainspring
    
    The spindle represents a pointer, like the hand of a clock,
    that can only point to one thing at a time. 
    It's defining characteristic is that, despite the 
    presence of a large number of active features
    at any one time, it must choose only one.
    
    The basis of choosing a feature is that feature's salience;
    the most salient feature is selected.
    Salience is very simple right now. A feature's salience is 
    determined only by the magnitude of its activity
    and how recently it was observed.
    
    TODO:
    As the spindle matures, there are many other factors 
    that might be considered in calculating a feature's salience.
    Here is a partial list:
    * tendency of a feature to be associated with reward or punishment,
    * whether it was recently salient, but not selected (salience decay),
    * rate of change of feature activity.
    
    It is also possible for the spindle to attend to short term memory
    features in the mainspring. This is yet to be implemented, but could
    serve as the basis for remembering short sequences of symbols or 
    instructions, often referred to as 'working memory'.

    In addition, the spindle can query the mainspring to determine likely 
    subsequent features, and attend to one of those. This provides
    a mechanism for multi-step planning and mental simulation.
    """
    def __init__(self, initial_size):
        self.num_cables = initial_size
        feature_shape = (self.num_cables, 1)
        self.INITIAL_TIME = 1e10
        self.time_since_seen = self.INITIAL_TIME * np.ones(feature_shape)

    def step(self, cable_activities):
        """ Pick a feature to attend to """
        self.time_since_seen += 1.
        recency_factor = 1. - 1. / self.time_since_seen
        salience = cable_activities * recency_factor
        winners = np.where(salience == np.max(salience))[0]
        attended_index = winners[np.random.randint(winners.size)]
        self.time_since_seen[attended_index] = 1.
        attended_activity = cable_activities[attended_index,0]
        return attended_index, attended_activity

    def add_cables(self, num_new_cables):
        """ Add new cables to the hub when new gearboxes are created """ 
        self.num_cables += num_new_cables
        features_shape = (self.num_cables, 1)
        self.time_since_seen = tools.pad(self.time_since_seen, features_shape,
                                val=self.INITIAL_TIME)

    def visualize(self):
        pass
