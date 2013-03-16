
from .base_world import World as BaseWorld
import numpy as np

class World(BaseWorld):
    """ grid_1D_noise.World
    One-dimensional grid task with noise. In this task, the agent steps forward and backward 
    along three positions on a line. The second position is rewarded and the first and third
    positions are punished. Also, any actions are penalized to a lesser degree. It also includes 
    some basic feature inputs that are pure noise. See Chapter 4 of the Users Guide for details.
    Optimal performance is a reward of between 90 per time step.
    """
    
    def __init__(self, lifespan=None):
        super(World, self).__init__()
        
        if lifespan is None:
            self.LIFESPAN = 10 ** 4
        else:
            self.LIFESPAN = lifespan
        self.REPORTING_PERIOD = 10 ** 4
        self.REWARD_MAGNITUDE = 100.
        self.ENERGY_COST = 0.01 * self.REWARD_MAGNITUDE
        self.JUMP_FRACTION = 0.1
        self.display_state = False  
        self.name = 'noisy one dimensional grid world'
        self.announce()

        self.num_real_sensors = 3

        """ Number of primitives that have no basis in the world. These are noise meant to distract. """
        self.num_noise_sensors = 15        
        self.num_sensors = self.num_noise_sensors + self.num_real_sensors
        self.num_actions = 3
        self.MAX_NUM_FEATURES = self.num_sensors + self.num_actions
        
        self.world_state = 0      
        self.simple_state = 0       


    def step(self, action): 
        if action is None:
            action = np.zeros(self.num_actions, 1)
        action = action.ravel()
        self.timestep += 1 
        step_size = action[0] - action[1]
                        
        """ An approximation of metabolic energy """
        energy = action[0] + action[1]
        self.world_state = self.world_state + step_size
        
        """ At random intervals, jump to a random position in the world """
        if np.random.random_sample() < self.JUMP_FRACTION:
            self.world_state = self.num_real_sensors * np.random.random_sample()

        """ Ensure that the world state falls between 0 and num_real_sensors """
        self.world_state -= self.num_real_sensors * \
                            np.floor_divide(self.world_state, self.num_real_sensors)
        self.simple_state = int(np.floor(self.world_state))
        
        """ Assign sensors as zeros or ones. 
        Represent the presence or absence of the current position in the bin.
        """
        real_sensors = np.zeros(self.num_real_sensors)
        real_sensors[self.simple_state] = 1

        """ Generate a set of noise sensors """
        noise_sensors = np.round(np.random.random_sample(self.num_noise_sensors))
        sensors = np.hstack((real_sensors, noise_sensors))

        reward = -self.REWARD_MAGNITUDE
        if self.simple_state == 1:
            reward = self.REWARD_MAGNITUDE
        reward -= energy * self.ENERGY_COST        
        self.display(action)
        return sensors, reward

    
    def set_agent_parameters(self, agent):
        #agent.perceiver.NEW_FEATURE_THRESHOLD = 1.0
        agent.reward_min = -100.
        agent.reward_max = 100.


    def display(self, action):
        if (self.display_state):
            state_image = ['.'] * (self.num_real_sensors + self.num_actions + 2)
            state_image[self.simple_state] = 'O'
            state_image[self.num_real_sensors:self.num_real_sensors + 2] = '||'

            action_index = np.nonzero(action)[0]
            if action_index.size > 0:
                state_image[self.num_real_sensors + 2 + action_index[0]] = 'x'
            print(''.join(state_image))
            
        if (self.timestep % self.REPORTING_PERIOD) == 0:
            print("world age is %s timesteps " % self.timestep)
