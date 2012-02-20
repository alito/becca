'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''
import random
import logging

import numpy as np

from .world import World

#import matplotlib.pyplot as plt

class Grid_1D_ms(World):
    ''' grid_1D_ms.World

    One-dimensional grid task, multi-step

    In this task, the agent steps forward and backward along a
    line. The fourth position is rewarded (1/2) and the ninth
    position is punished (-1/2).

    This is intended to be as similar as possible to the 
    one-dimensional grid task, but require multi-step planning for optimal 
    behavior.

    Optimal performance is between 0.25 and 0.3 reward per time step.

    '''

    def __init__(self):
        ''' default constructor
        '''

        super(Grid_1D_ms, self).__init__()
        
        self.num_sensors = 1
        self.num_primitives = 9
        self.num_actions = 3

        self.world_state = 0            

        self.REPORTING_PERIOD = 10 ** 3
        self.LIFESPAN = 10 ** 4
        
    
    def display(self):
        ''' provides an intuitive display of the current state of the World 
        to the user
        '''
        if (self.display_features):
            state_image = ['.'] * self.num_primitives
            state_image[self.world_state] = 'O'
            logging.info('world timestep %s    %s' % (self.timestep, ''.join(state_image)))
            
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            logging.info("%s timesteps done" % self.timestep)
            self.record_reward_history()
            self.cumulative_reward = 0
            self.show_reward_history()


        
    def step(self, action): 
        ''' advances the World by one timestep.
        '''
        self.timestep += 1 
        action = np.round(action)

        if random.random() < 0.1:
            action += round(random.random() * 6) * np.round(np.random.random_sample(3))

        energy = action[0] + action[1]
        
        self.world_state += action[0] - action[1]
        
        # ensures that the world state falls between 0 and 9
        self.world_state -= 9 * np.floor_divide(self.world_state, self.num_primitives)
        simple_state = int(self.world_state)
        
        # Assigns basic_feature_input elements as binary. Represents the presence
        # or absence of the current position in the bin.
        sensors = np.zeros(self.num_sensors)
        primitives = np.zeros(self.num_primitives)
        primitives[simple_state] = 1
        
        # Assigns reward based on the current state
        reward = primitives[8] * (-0.5)
        reward += primitives[3] * ( 0.5)
        
        # Punishes actions just a little.
        reward -= energy / 100
        reward = np.max(reward, -1)
        
        self.log(sensors, primitives, reward)
        self.display()

        return sensors, primitives, reward
        
        
    def final_performance(self):
        ''' When the world terminates, this returns the average performance 
        of the agent on the last 3 blocks.
        '''
        if (self.timestep > self.LIFESPAN):
            performance = np.mean(self.reward_history[-3:]) / self.REPORTING_PERIOD
            #plt.ioff()
            #plt.show()
            
            assert(performance >= -1.)
            return performance
        
        return -2
