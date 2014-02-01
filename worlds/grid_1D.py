import numpy as np

from worlds.base_world import World as BaseWorld

class World(BaseWorld):
    """ 
    One-dimensional grid task
    
    In this task, the agent steps forward and backward along 
    a nine-position line. The fourth position is rewarded and 
    the ninth position is punished. There is also a slight 
    punishment for effort expended in trying to move, 
    i.e. taking actions. This is intended to be a simple-as-possible 
    task for troubleshooting BECCA. See Chapter 4 of the 
    Users Guide for details.
    Optimal performance is a reward of about 90 per time step.
    """
    def __init__(self, lifespan=None):
        """ Set up the world """
        BaseWorld.__init__(self, lifespan)
        self.VISUALIZE_PERIOD = 10 ** 4
        self.REWARD_MAGNITUDE = 100.
        self.ENERGY_COST =  self.REWARD_MAGNITUDE / 100.
        self.JUMP_FRACTION = 0.1
        self.name = 'grid_1D_a'
        self.name_long = 'one dimensional grid world'
        print "Entering", self.name_long
        self.num_sensors = 9
        self.num_actions = 9
        self.action = np.zeros((self.num_actions,1))
        self.world_state = 0
        self.simple_state = 0
        self.display_state = True 
    
    def step(self, action): 
        """ Take one time step through the world """
        self.action = action
        self.timestep += 1 
        # Find the step size as combinations of the action commands
        step_size = (self.action[0] + 
                 2 * self.action[1] + 
                 3 * self.action[2] + 
                 4 * self.action[3] - 
                     self.action[4] - 
                 2 * self.action[5] - 
                 3 * self.action[6] - 
                 4 * self.action[7])
        # Action cost is an approximation of metabolic energy
        self.energy=(self.action[0] + 
                 2 * self.action[1] + 
                 3 * self.action[2] + 
                 4 * self.action[3] + 
                     self.action[4] + 
                 2 * self.action[5] + 
                 3 * self.action[6] + 
                 4 * self.action[7])
        self.world_state = self.world_state + step_size        
        # At random intervals, jump to a random position in the world
        if np.random.random_sample() < self.JUMP_FRACTION:
	    self.world_state = self.num_sensors * np.random.random_sample()
        # Ensure that the world state falls between 0 and 9
        self.world_state -= self.num_sensors * np.floor_divide(
                self.world_state, self.num_sensors)
        self.simple_state = int(np.floor(self.world_state))
        if self.simple_state == 9:
            self.simple_state = 0
        # Assign basic_feature_input elements as binary. 
        # Represent the presence or absence of the current position in the bin.
        sensors = np.zeros(self.num_sensors)
        sensors[self.simple_state] = 1
        reward = self.assign_reward(sensors)
        return sensors, reward

    def assign_reward(self, sensors):
        """ Assign reward based on the current state """
        reward = 0.
        reward -= sensors[8] * self.REWARD_MAGNITUDE
        reward += sensors[3] * self.REWARD_MAGNITUDE
        # Punish actions just a little
        reward -= self.energy  * self.ENERGY_COST
        reward = np.maximum(reward, -self.REWARD_MAGNITUDE)
        return reward
        
    def set_agent_parameters(self, agent):
        """ Turn a few of the knobs to adjust BECCA for this world """
        # Prevent the agent from forming any groups
        agent.reward_min = -100.
        agent.reward_max = 100.
        
    def visualize(self, agent):
        """ Show what's going on in the world """
        if (self.display_state):
            state_image = ['.'] * (self.num_sensors + self.num_actions + 2)
            state_image[self.simple_state] = 'O'
            state_image[self.num_sensors:self.num_sensors + 2] = '||'
            action_index = np.where(self.action > 0.1)[0]
            if action_index.size > 0:
                for i in range(action_index.size):
                    state_image[self.num_sensors + 2 + action_index[i]] = 'x'
            print(''.join(state_image))
            
        if (self.timestep % self.VISUALIZE_PERIOD) == 0:
            print("world age is %s timesteps " % self.timestep)
