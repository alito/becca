'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''
import random
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

        self.sensors = np.zeros(self.num_sensors)
        self.primitives = np.zeros(self.num_primitives)
        self.actions = np.zeros(self.num_actions)

        self.world_state = 0            

        self.REPORTING_PERIOD = 10 ** 3
        self.LIFESPAN = 10 ** 4
        
    
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
        Accepts agent as an argument only so that it can occasionally backup
        the agent's state to disk.
        '''
        self.timestep += 1 
        action = np.round(action)

        if random.random() < 0.1:
            action += round(random.random() * 6) * np.round(np.random.random_sample((3,)))

        energy = action[0] + action[1]
        
        self.world_state += action[0] - action[1]
        
        # ensures that the world state falls between 0 and 9
        self.world_state -= 9 * np.floor_divide(self.world_state, self.num_primitives)
        simple_state = int(self.world_state)
        
        # Assigns basic_feature_input elements as binary. Represents the presence
        # or absence of the current position in the bin.
        self.sensors = np.zeros(self.num_sensors)
        self.primitives = np.zeros(self.num_primitives)
        self.primitives[simple_state] = 1
        
        # Assigns reward based on the current state
        self.reward = self.primitives[8] * (-0.5)
        self.reward += self.primitives[3] * ( 0.5)
        
        # Punishes actions just a little.
        self.reward -= energy / 100
        self.reward = np.max( self.reward, -1)
        
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
