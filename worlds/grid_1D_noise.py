
from .base_world import World as BaseWorld

import numpy as np

class World(BaseWorld):
    """ grid_1D_noise.World
    One-dimensional grid task with noise

    In this task, the agent steps forward and backward along three positions 
    on a line. The second position is rewarded (1/2) and the first and third
    positions are punished (-1/2). Also, any actions are penalized (-1/10).
    It also includes some basic feature inputs that are pure noise.

    Optimal performance is between 0.3 and 0.35 reward per time step.
    """
    
    def __init__(self):
                
        super(World, self).__init__()
        
        self.REPORTING_PERIOD = 10 ** 3
        self.LIFESPAN = 2 * 10 ** 4
        self.REWARD_MAGNITUDE = 1.
        self.ENERGY_COST = 0.1      
        self.display_state = False  
        self.name = 'noisy one dimensional grid world'
        self.announce()


        """ Number of primitives that have no basis in the world. These
        are noise meant to distract. """
        self.num_sensors = 0
        self.num_real_primitives = 3
        self.num_noise_primitives = 12
        self.num_primitives = self.num_noise_primitives + \
                                self.num_real_primitives
        self.num_actions = 3

        self.sensors = np.zeros(self.num_sensors)
        
        self.world_state = 0      
        self.simple_state = 0       


    def step(self, action): 
        """ Advance the World by one timestep """

        if action is None:
            action = np.zeros(self.num_actions, 1)
        action = action.ravel()
        
        self.timestep += 1 

        step_size = action[0] - action[1]
                        
        """ An approximation of metabolic energy """
        energy = action[0] + action[1]

        self.world_state = self.world_state + step_size
        
        """ Ensure that the world state falls between 0 and 
        num_real_primitives. 
        """
        self.world_state -= self.num_real_primitives * \
                            np.floor_divide(self.world_state, 
                                            self.num_real_primitives)
        self.simple_state = int(np.floor(self.world_state))
        

        """ Assign primitives as zeros or ones. 
        Represent the presence or absence of the current position in the bin.
        """
        real_primitives = np.zeros(self.num_real_primitives)
        real_primitives[self.simple_state] = 1

        """ Generate a set of noise primitives """
        noise_primitives = np.round(np.random.random_sample
                                    (self.num_noise_primitives))
        primitives = np.hstack((real_primitives, noise_primitives))

        reward = -self.REWARD_MAGNITUDE
        if self.simple_state == 1:
            reward = self.REWARD_MAGNITUDE

        """ Punish actions just a little """
        reward -= energy * self.ENERGY_COST
        
        self.display(action)

        return self.sensors, primitives, reward

    
    def set_agent_parameters(self, agent):
        """ Prevent the agent from forming any groups """
        agent.perceiver.NEW_FEATURE_THRESHOLD = 1.0


    def display(self, action):
        """ Provide an intuitive display of the current state of the World 
        to the user.
        """
        if (self.display_state):
            state_image = ['.'] * (self.num_real_primitives + self.num_actions + 2)
            state_image[self.simple_state] = 'O'
            state_image[self.num_real_primitives:self.num_real_primitives + 2] = '||'

            action_index = np.nonzero(action)[0]
            if action_index.size > 0:
                state_image[self.num_real_primitives + 2 + action_index[0]] = 'x'

            print(''.join(state_image))
            
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            print("world age is %s timesteps " % self.timestep)
