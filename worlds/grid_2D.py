'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''
import numpy as np
#import matplotlib.pyplot as plt

from .world import World

class Grid_2D(World):
    ''' grid_2D.World
    Two-dimensional grid task

    In this task, the agent steps North, South, East, or West in a
    5 x 5 grid-world. Position (4,4) is rewarded (1/2) and (2,2) is
    punished (-1/2). There is also a penalty of -1/20 for each horizontal
    or vertical step taken.
    Horizonal and vertical positions are reported
    separately as basic features, rather than raw sensory inputs.

    This is intended to be a
    simple-as-possible-but-slightly-more-interesting-
    that-the-one-dimensional-task task for troubleshooting BECCA.

    Optimal performance is between 0.3 and 0.35 reward per time step.
    '''

    def __init__(self):
        ''' default constructor
        '''

        super(Grid_2D,self).__init__()
        
        self.REPORTING_PERIOD = 10 ** 3
        self.LIFESPAN = 10 ** 4
        self.ENERGY_PENALTY = 0.05

        self.num_sensors = 1
        self.num_actions = 9            
        self.world_size = 5
        self.num_primitives = self.world_size ** 2
        self.world_state = np.array([1, 1])
        self.simple_state = self.world_state.copy()

        self.target = (3,3)
        self.obstacle = (1,1)

        self.sensors = np.zeros(self.num_sensors)
        self.primitives = np.zeros(self.num_primitives)
        self.actions = np.zeros(self.num_actions)

		
    
    def display(self):
        ''' provides an intuitive display of the current state of the World 
        to the user
        '''
        if (self.display_features):
            state_img = ['.'] * self.num_primitives
            state_img[self.world_state] = 'O'
            print('world timestep ' + str(self.timestep) + '  ' + ''.join(state_img))
            
        if (np.mod(self.timestep, self.REPORTING_PERIOD) == 0):
            self.reward_history = np.append(self.reward_history, self.cumulative_reward)
            self.cumulative_reward = 0
            #plt.plot(self.reward_history)


        
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

        self.basic_feature_input = np.zeros((self.num_primitives,))
        self.basic_feature_input[self.simple_state[1] + self.simple_state[0] * self.world_size] = 1

        self.reward = 0
        if tuple(self.simple_state.flatten()) == self.obstacle:
            self.reward = -0.5
        elif tuple(self.simple_state.flatten()) == self.target:
            self.reward = 0.5

        self.reward -= self.ENERGY_PENALTY * energy

        
        self.log()
        self.display()
        

        
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
