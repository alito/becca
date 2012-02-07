'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''

import numpy as np

class Agent(object):
    '''A general reinforcement learning agent, modeled on human neurophysiology 
    and performance.  It takes in a time series of 
    sensory input vectors and a scalar reward and puts out a time series 
    of motor commands.  
    New features are created as necessary to adequately represent the data.
    '''

    def __init__(self, num_sensors, num_primitives, num_actions):
        '''
        Constructor
        '''
        self.num_sensors = num_sensors
        self.num_primitives = num_primitives
        self.num_actions = num_actions

        self.actions = np.zeros(self.num_actions)
        
        self.timestep = 0
        
    
    def step(self, sensors, primitives, reward):
        self.sensors = sensors.copy()
        self.primitives = primitives.copy()
        self.reward = reward

        self.timestep += 1   

        self.actions = np.zeros(self.num_actions)
        self.actions[np.random.randint(self.num_actions)] = 1

        # debug
        """ 
        print('========================')
        print('wm:');
        self.working_memory.display()
        print('fa:');
        self.feature_activity.display()
        print('ra:');
        reactive_action.display()
        print('da:');
        self.planner.action.display()
        print('aa:');
        self.action.display()
        """
        
        return()
