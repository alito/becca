'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''
import logging

import numpy as np
#import matplotlib.pyplot as plt

from .world import World

class Grid_2D_dc(World):
    ''' grid_2D_dc.World
    Two-dimensional grid task

    dc stands for decoupled. It's just like the task_grid_2D task except that 
    the two dimensions are decoupled. The basic feature vector represents a
    row and a column separately, not coupled together.

    In this task, the agent steps North, South, East, or West in a
    5 x 5 grid-world. Position (4,4) is rewarded (1/2) and (2,2) is
    punished (-1/2).  There is also a penalty of -1/20 for each horizontal
    or vertical step taken. Horizonal and vertical positions are reported
    separately as basic features, rather than raw sensory inputs.

    This is intended to be a
    simple-as-possible-but-slightly-more-interesting-
    that-the-one-dimensional-task task for troubleshooting BECCA.

    Optimal performance is between 0.3 and 0.35.

    '''

    def __init__(self, graphs=True):
        ''' default constructor
        '''

        super(Grid_2D_dc,self).__init__(graphs=graphs)
        
        self.REPORTING_PERIOD = 10 ** 3
        self.LIFESPAN = 10 ** 4
        self.ENERGY_PENALTY = 0.05

        self.num_sensors = 1
        self.num_actions = 9            
        self.world_size = 5
        self.num_primitives = self.world_size * 2
        self.world_state = np.array([1, 1])
        self.simple_state = self.world_state.copy()

        self.target = (3,3)
        self.obstacle = (1,1)

        self.sensors = np.zeros(self.num_sensors)


        self.motor_output_history = np.array([])            

    
    def display(self):
        ''' provides an intuitive display of the current state of the World 
        to the user
        '''            
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

        self.world_state += (action[0:2] + 2 * action[2:4] - action[4:6] - 2 * action[6:8]).transpose()

        energy = np.sum(action[0:2]) + np.sum(2 * action[2:4]) + np.sum(action[4:6]) - np.sum(2 * action[6:8])
        

        #enforces lower and upper limits on the grid world by looping them around.
        #It actually has a toroidal topology.
        indices = self.world_state >= self.world_size - 0.5
        self.world_state[indices] -= self.world_size

        indices = self.world_state <= -0.5
        self.world_state[indices] += self.world_size

        self.simple_state = np.round(self.world_state)

        primitives = np.zeros((self.num_primitives,))
        primitives[self.simple_state[0]] = 1
        primitives[self.simple_state[1] + self.world_size] = 1

        reward = 0
        if tuple(self.simple_state.flatten()) == self.obstacle:
            reward = -0.5
        elif tuple(self.simple_state.flatten()) == self.target:
            reward = 0.5

        reward -= self.ENERGY_PENALTY * energy

        
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
            
            assert(performance >= -1.)
            return performance
        
        return -2