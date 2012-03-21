'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''
import logging

import numpy as np

from .world import World

#import matplotlib.pyplot as plt

class Grid_1D_noise(World):
    ''' grid_1D_noise.World
    One-dimensional grid task with noise

    In this task, the agent steps forward and backward along three positions 
    on a line. The second position is rewarded (1/2) and the first and third
    positions are punished (-1/2). Also, any actions are penalized (-1/10).
    It also includes some basic feature inputs that are pure noise.

    Optimal performance is between 0.3 and 0.35 reward per time step.
    '''
    
    def __init__(self, graphs=True):
        ''' default constructor
        '''

        super(Grid_1D_noise, self).__init__(graphs=graphs)
        
        self.noise_inputs = 7
        self.num_sensors = 1
        self.num_real_features = 3
        self.num_actions = 3
        self.num_primitives = self.noise_inputs + self.num_real_features

        self.sensors = np.zeros(self.num_sensors)
        
        self.world_state = 0            

        self.REPORTING_PERIOD = 10 ** 3
        self.LIFESPAN = 10 ** 4
        self.ENERGY_PENALTY = 0.1        

        
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
        energy = np.sum(action)

        self.world_state += action[0] - action[1]
        self.world_state = np.round(self.world_state)
        self.world_state = min(self.world_state, 2)
        self.world_state = max(self.world_state, 0)

        real_features = np.zeros(self.num_real_features)
        real_features[int(self.world_state)] = 1

        noise = np.round(np.random.random_sample(self.noise_inputs))
        primitives = np.hstack((real_features, noise))

        reward = -0.5
        if int(self.world_state) == 2:
            reward = 0.5

        reward -= energy * self.ENERGY_PENALTY
        
        self.log(self.sensors, primitives, reward)
        self.display()

        return self.sensors, primitives, reward

    
    def final_performance(self):
        ''' When the world terminates, this returns the average performance 
        of the agent on the last 3 blocks.
        '''
        if (self.timestep > self.LIFESPAN):
            performance = np.mean(self.reward_history[-3:]) / self.REPORTING_PERIOD
            #plt.ioff()
            #plt.show()
            
            return performance
        
        return -2
