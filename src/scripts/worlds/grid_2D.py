import numpy as np

from worlds.base_world import World as BaseWorld

class World(BaseWorld):
    """ 
    Two-dimensional grid task

    In this task, the agent steps North, South, East, or West in 
    a 5 x 5 grid-world. Position (4,4) is rewarded and (2,2) 
    is punished. There is also a lesser penalty for each 
    horizontal or vertical step taken. Horizonal and vertical 
    positions are reported separately as basic features, 
    rather than raw sensory inputs.
    See Chapter 4 of the Users Guide for details.  
    Optimal performance is a reward of about 90 per time step.
    """
    def __init__(self, lifespan=None):
        """ Set up the world """
        BaseWorld.__init__(self, lifespan)
        self.VISUALIZE_PERIOD = 10 ** 4
        self.REWARD_MAGNITUDE = 100.
        self.ENERGY_COST = 0.05 * self.REWARD_MAGNITUDE
        self.JUMP_FRACTION = 0.1
        self.display_state = True
        self.name = 'two dimensional grid world'
        print "Entering", self.name
        self.num_actions = 9            
        self.world_size = 5
        self.num_sensors = self.world_size ** 2
        self.world_state = np.array([1, 1])
        self.simple_state = self.world_state.copy()
        self.target = (3,3)
        self.obstacle = (1,1)
    
    def step(self, action): 
        """ Take one time step through the world """
        self.action = action.ravel()
        self.timestep += 1
        self.world_state += (self.action[0:2] - 
                             self.action[4:6] + 
                             2 * self.action[2:4] -
                             2 * self.action[6:8]).T
        energy = (np.sum(self.action[0:2]) + 
                  np.sum(self.action[4:6]) + 
                  np.sum(2 * self.action[2:4]) +
                  np.sum(2 * self.action[6:8]))
        # At random intervals, jump to a random position in the world
        if np.random.random_sample() < self.JUMP_FRACTION:
            self.world_state = np.random.random_integers(
                    0, self.world_size, self.world_state.shape)
        # Enforce lower and upper limits on the grid world 
        # by looping them around
        indices = (self.world_state >= self.world_size - 0.5).nonzero()
        self.world_state[indices] -= self.world_size
        indices = (self.world_state <= -0.5).nonzero()
        self.world_state[indices] += self.world_size
        self.simple_state = np.round(self.world_state)
        sensors = self.assign_sensors()
        reward = 0
        if tuple(self.simple_state.flatten()) == self.obstacle:
            reward = - self.REWARD_MAGNITUDE
        elif tuple(self.simple_state.flatten()) == self.target:
            reward = self.REWARD_MAGNITUDE
        reward -= self.ENERGY_COST * energy
        return sensors, reward
    
    def assign_sensors(self):
        """ Construct the sensor array from the state information """
        sensors = np.zeros(self.num_sensors)
        sensors[self.simple_state[1] + 
                self.simple_state[0] * self.world_size] = 1
        return sensors
    
    def set_agent_parameters(self, agent):
        """ Set a few parameters in the agent """
        # Prevent the agent from forming any groups
        agent.reward_min = -100.
        agent.reward_max = 100.
        return

    def visualize(self, agent):
        """ Show the state of the world and the agent """
        if (self.display_state):
            print ''.join(['state', str(self.simple_state), '  action', 
                           str((self.action[0:2] + 2 * self.action[2:4] - 
                                self.action[4:6] - 2 * self.action[6:8]).T)])
        if (self.timestep % self.VISUALIZE_PERIOD) != 0:
            return
        
        print("world age is %s timesteps " % self.timestep)
        agent.visualize()
        projections = agent.get_projections(to_screen=True)
        return
