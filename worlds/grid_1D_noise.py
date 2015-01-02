"""
One-dimensional grid task with noise

In this task, the agent has the challenge of discriminating between
actual informative state sensors, and a comparatively large number
of sensors that are pure noise distractors. Many learning methods
make the implicit assumption that all sensors are informative.
This task is intended to break them.
"""
import numpy as np
from base_world import World as BaseWorld

class World(BaseWorld):
    """ 
    One-dimensional grid world with noise
    
    In this world, the agent steps forward and backward 
    along three positions on a line. The second position is rewarded 
    and the first and third positions are punished. Also, any actions 
    are penalized somewhat. It also includes some inputs that are pure noise. 
    Optimal performance is a reward of between 90 per time step.
    """
    
    def __init__(self, lifespan=None):
        """ Set up the world """
        BaseWorld.__init__(self, lifespan)
        self.VISUALIZE_PERIOD = 10 ** 4
        self.REWARD_MAGNITUDE = 100.
        self.ENERGY_COST = 0.01 * self.REWARD_MAGNITUDE
        self.JUMP_FRACTION = 0.1
        self.display_state = True  
        self.name = 'grid_1D_noise'
        self.name_long = 'noisy one dimensional grid world'
        print "Entering", self.name_long
        self.num_real_sensors = 3
        # Number of sensors that have no basis in the world. 
        # These are noise meant to distract.
        self.num_noise_sensors = 15        
        self.num_sensors = self.num_noise_sensors + self.num_real_sensors
        self.num_actions = 3
        self.action = np.zeros((self.num_actions,1))
        self.world_state = 0      
        self.simple_state = 0       

    def step(self, action): 
        """ Take one time step through the world """
        self.action = action.copy().ravel()
        self.timestep += 1 
        step_size = self.action[0] - self.action[1]
        # An approximation of metabolic energy
        energy = self.action[0] + self.action[1]
        self.world_state = self.world_state + step_size
        # At random intervals, jump to a random position in the world
        if np.random.random_sample() < self.JUMP_FRACTION:
            self.world_state = (self.num_real_sensors * 
                                np.random.random_sample())
        # Ensure that the world state falls between 0 and num_real_sensors 
        self.world_state -= (self.num_real_sensors * 
                             np.floor_divide(self.world_state, 
                                             self.num_real_sensors))
        self.simple_state = int(np.floor(self.world_state))
        # Assign sensors as zeros or ones. 
        # Represent the presence or absence of the current position in the bin.
        real_sensors = np.zeros(self.num_real_sensors)
        real_sensors[self.simple_state] = 1
        # Generate a set of noise sensors
        noise_sensors = np.round(np.random.random_sample(
                self.num_noise_sensors))
        sensors = np.hstack((real_sensors, noise_sensors))
        reward = -self.REWARD_MAGNITUDE
        if self.simple_state == 1:
            reward = self.REWARD_MAGNITUDE
        reward -= energy * self.ENERGY_COST        
        return sensors, reward
    
    def set_agent_parameters(self, agent):
        """ Make some adjustements, as necessary, to the agent """
        pass

    def visualize(self, agent):
        """ Show what's going on in the world """
        if (self.display_state):
            state_image = ['.'] * (self.num_real_sensors + 
                                   self.num_actions + 2)
            state_image[self.simple_state] = 'O'
            state_image[self.num_real_sensors:self.num_real_sensors + 2] = '||'
            action_index = np.where(self.action > 0.1)[0]
            if action_index.size > 0:
                state_image[self.num_real_sensors + 2 + action_index[0]] = 'x'
            print(''.join(state_image))

        if (self.timestep % self.VISUALIZE_PERIOD) == 0:
            print("world age is %s timesteps " % self.timestep)
