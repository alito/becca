import numpy as np

from .base_world import World as BaseWorld

class World(BaseWorld):
    """
    One-dimensional grid task, multi-step variation

    In this task, the agent steps forward and backward along a line. 
    The fourth position is rewarded and the ninth position is punished. 
    This is intended to be as similar as possible to the 
    one-dimensional grid task, but require multi-step planning 
    for optimal behavior.
    See Chapter 4 of the Users Guide for details.
    Optimal performance is a reward of about 85 per time step.
    """

    def __init__(self, lifespan=None):
        """ Set up the world """
        BaseWorld.__init__(self, lifespan)
        self.VISUALIZE_PERIOD = 10 ** 4
        self.REWARD_MAGNITUDE = 100.
        self.ENERGY_COST = 0.01 * self.REWARD_MAGNITUDE
        self.JUMP_FRACTION = 0.10
        self.display_state = True 
        self.name = 'grid_1D_ms'
        self.name_long = 'multi-step one dimensional grid world'
        print "Entering", self.name_long
        self.num_sensors = 9
        self.num_actions = 3
        self.action = np.zeros((self.num_actions,1))
        self.world_state = 0            
        self.simple_state = 0
            
    def step(self, action): 
        """ Take one time step through the world """
        self.action = action.ravel()
        self.timestep += 1 
        energy = self.action[0] + self.action[1]
        self.world_state += self.action[0] - self.action[1]
        # Occasionally add a perturbation to the action to knock it 
        # into a different state 
        if np.random.random_sample() < self.JUMP_FRACTION:
            self.world_state = self.num_sensors * np.random.random_sample()
        # Ensure that the world state falls between 0 and 9
        self.world_state -= self.num_sensors * np.floor_divide(
                self.world_state, self.num_sensors)
        self.simple_state = int(np.floor(self.world_state))
        # TODO do this more elegantly
        if self.simple_state == 9:
            self.simple_state = 0
        # Assign sensors as zeros or ones. 
        # Represent the presence or absence of the current position in the bin.
        sensors = np.zeros(self.num_sensors)
        sensors[self.simple_state] = 1
        # Assign reward based on the current state 
        reward = sensors[8] * (-self.REWARD_MAGNITUDE)
        reward += sensors[3] * (self.REWARD_MAGNITUDE)
        # Punish actions just a little 
        reward -= energy * self.ENERGY_COST
        reward = np.max(reward, -1)
        return sensors, reward

    def set_agent_parameters(self, agent):
        """ Turn a few of the knobs to adjust BECCA for this world """
        agent.reward_min = -100.
        agent.reward_max = 100.

    def visualize(self, agent):
        """ Provide an intuitive display of the current state of the World """
        if (self.display_state):
            state_image = ['.'] * self.num_sensors
            state_image[self.simple_state] = 'O'
            print(''.join(state_image))
            
        if (self.timestep % self.VISUALIZE_PERIOD) == 0:
            print("world age is %s timesteps " % self.timestep)
