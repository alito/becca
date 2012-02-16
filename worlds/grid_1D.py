'''
Created on Jan 11, 2012

@author: brandon_rohrer
'''

import numpy as np

from .world import World
#import matplotlib.pyplot as plt

class Grid_1D(World):
    ''' grid_1D.World
    One-dimensional grid task

    In this task, the agent steps forward and backward along a
    line. The fourth position is rewarded (+1/2) and the ninth
    position is punished (-1/2). There is also a slight punishment
    for effort expended in trying to move, i.e. taking actions.
    
    This is intended to be a simple-as-possible task for
    troubleshooting BECCA.
    
    The theoretically optimal performance without exploration is 0.5 
    reward per time step.
    In practice, the best performance the algorithm can achieve with the 
    exploration levels given is around 0.35 to 0.37 reward per time step.
    '''

    def __init__(self):
        ''' default constructor
        '''

        super(Grid_1D, self).__init__()
        
        self.num_sensors = 1
        self.num_primitives = 9
        self.num_actions = 9

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

        step_size = (action[0] + 
                 2 * action[1] + 
                 3 * action[2] + 
                 4 * action[3] - 
                     action[4] - 
                 2 * action[5] - 
                 3 * action[6] - 
                 4 * action[7])
                        
        # an approximation of metabolic energy
        energy    = (action[0] + 
                 2 * action[1] + 
                 3 * action[2] + 
                 4 * action[3] + 
                     action[4] + 
                 2 * action[5] + 
                 3 * action[6] + 
                 4 * action[7])

        self.world_state = self.world_state + step_size
        
        # ensures that the world state falls between 0 and 9
        self.world_state = (self.world_state - 
                            9 * np.floor_divide(self.world_state, self.num_primitives))
        simple_state = int(np.floor(self.world_state))
        
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
