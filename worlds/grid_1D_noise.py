
import numpy as np
from .base_world import World as BaseWorld

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
        self.REWARD_MAGNITUDE = 0.5
        self.ENERGY_COST = 0.1      
        self.display_state = False  

        """ Number of primitives that have no basis in the world. These
        are noise meant to distract. """
        self.num_sensors = 1
        self.num_real_primitives = 3
        self.num_noise_primitives = 3
        self.num_primitives = self.num_noise_primitives + \
                                self.num_real_primitives
        self.num_actions = 3

        self.sensors = np.zeros(self.num_sensors)
        
        self.world_state = 0      
        self.simple_state = 0       


    def step(self, action): 
        """ Advance the World by one timestep """

        if action is None:
            action = np.zeros(self.num_actions)

        action = np.round(action)
        
        self.timestep += 1

        energy = np.sum(action)

        """ Ensure that the world state falls between 0 and 9 """
        self.world_state -= self.num_real_primitives * \
                            np.floor_divide(self.world_state, 
                                            self.num_real_primitives)
        self.simple_state = int(np.floor(self.world_state))

        self.world_state += action[0] - action[1]
        self.world_state = np.round(self.world_state)
        
        """ Ensure that the world state falls between 0 and 2 """
        self.world_state = min(self.world_state, 2)
        self.world_state = max(self.world_state, 0)

        """ Assign primitives as zeros or ones. 
        Represent the presence or absence of the current position in the bin.
        """
        real_primitives = np.zeros(self.num_real_primitives)
        real_primitives[int(self.world_state)] = 1

        """ Generate a set of noise primitives """
        noise_primitives = np.round(np.random.random_sample
                                    (self.num_noise_primitives))
        primitives = np.hstack((real_primitives, noise_primitives))

        reward = -self.REWARD_MAGNITUDE
        if int(self.world_state) == 1:
            reward = self.REWARD_MAGNITUDE

        reward -= energy * self.ENERGY_COST
        
        self.display()

        return self.sensors, primitives, reward

    
    def set_agent_parameters(self, agent):
        """ Prevent the agent from forming any groups """
        agent.perceiver.NEW_GROUP_THRESHOLD = 1.0


    def display(self):
        """ Provide an intuitive display of the current state of the World 
        to the user.
        """
        if (self.display_state):
            state_image = ['.'] * self.num_real_primitives
            state_image[self.simple_state] = 'O'
            print(''.join(state_image))
            
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            print("world age is %s timesteps " % self.timestep)
